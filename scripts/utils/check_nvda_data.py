import duckdb

# Connect to the database
conn = duckdb.connect("data/db_backup/financial_data.duckdb", read_only=True)

# Check tables
print("Tables:")
print(conn.execute("SHOW TABLES").fetchdf())

# Check NVDA data
print("\nNVDA Data:")
try:
    print(
        conn.execute(
            "SELECT * FROM time_series_metrics WHERE ticker = 'NVDA' AND metric_name = 'Revenue' LIMIT 5"
        ).fetchdf()
    )
except Exception as e:
    print(f"Error querying NVDA data: {e}")

# Check if NVDA exists in companies table
print("\nNVDA in companies table:")
try:
    print(conn.execute("SELECT * FROM companies WHERE ticker = 'NVDA'").fetchdf())
except Exception as e:
    print(f"Error querying companies table: {e}")

# Close the connection
conn.close()
