import os

import duckdb

# Check if the database file exists
db_path = "data/db_backup/improved_financial_data.duckdb"
if not os.path.exists(db_path):
    print(f"Database file not found at {db_path}")
    exit(1)

try:
    # Connect to the database
    conn = duckdb.connect(db_path, read_only=True)

    # Check if the filings table exists
    tables = conn.execute("SHOW TABLES").fetchdf()
    print(f"Tables in database: {tables['name'].tolist()}")

    if "filings" in tables["name"].tolist():
        # Get total filings count
        total_filings = conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
        print(f"Total filings: {total_filings}")

        # Check if there are any NVDA filings
        nvda_filings = conn.execute("SELECT COUNT(*) FROM filings WHERE ticker = 'NVDA'").fetchone()[0]
        print(f"NVDA filings: {nvda_filings}")

        # Get a list of all tickers
        tickers = conn.execute("SELECT DISTINCT ticker FROM filings ORDER BY ticker").fetchdf()
        print(f"Tickers in database: {tickers['ticker'].tolist()}")
    else:
        print("No filings table found in the database")

    # Close the connection
    conn.close()
except Exception as e:
    print(f"Error: {e}")
