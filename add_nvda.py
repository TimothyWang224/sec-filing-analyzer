import datetime
import uuid

import duckdb

# Connect to the database
conn = duckdb.connect("data/db_backup/financial_data.duckdb", read_only=False)

# Add NVDA to companies table if it doesn't exist
try:
    # Check if NVDA already exists
    nvda_exists = conn.execute("SELECT COUNT(*) FROM companies WHERE ticker = 'NVDA'").fetchone()[0]

    if nvda_exists == 0:
        conn.execute(
            """
        INSERT INTO companies (ticker, name, created_at)
        VALUES ('NVDA', 'NVIDIA Corporation', ?)
        """,
            [datetime.datetime.now()],
        )
        print("Added NVDA to companies table")
    else:
        print("NVDA already exists in companies table")
except Exception as e:
    print(f"Error with companies table: {e}")

# Generate a unique ID for the filing
filing_id = str(uuid.uuid4())

# Add a sample filing for NVDA
try:
    conn.execute(
        """
    INSERT INTO filings (id, ticker, filing_type, filing_date, fiscal_year, fiscal_quarter, accession_number, created_at)
    VALUES (?, 'NVDA', '10-K', '2023-02-24', 2023, 4, '0000-12345-23-000123', ?)
    """,
        [filing_id, datetime.datetime.now()],
    )
    print(f"Added NVDA filing to filings table with ID: {filing_id}")
except Exception as e:
    print(f"Error adding NVDA filing: {e}")
    filing_id = None

# Add revenue data for NVDA
if filing_id:
    try:
        conn.execute(
            """
        INSERT INTO time_series_metrics (ticker, metric_name, fiscal_year, fiscal_quarter, value, unit, filing_id, created_at)
        VALUES ('NVDA', 'Revenue', 2023, 4, 26974000000, 'USD', ?, ?)
        """,
            [filing_id, datetime.datetime.now()],
        )
        print("Added NVDA revenue data to time_series_metrics table")
    except Exception as e:
        print(f"Error adding NVDA revenue data: {e}")

# Commit changes
conn.commit()

# Verify the data was added
print("\nVerifying NVDA in companies table:")
print(conn.execute("SELECT * FROM companies WHERE ticker = 'NVDA'").fetchdf())

print("\nVerifying NVDA filing:")
print(conn.execute("SELECT * FROM filings WHERE ticker = 'NVDA'").fetchdf())

print("\nVerifying NVDA revenue data:")
print(conn.execute("SELECT * FROM time_series_metrics WHERE ticker = 'NVDA' AND metric_name = 'Revenue'").fetchdf())

# Close the connection
conn.close()
