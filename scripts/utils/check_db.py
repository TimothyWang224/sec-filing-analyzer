import duckdb

# Connect to the database
conn = duckdb.connect("data/db_backup/financial_data.duckdb", read_only=True)

# Check companies
print("Companies:")
print(conn.execute("SELECT * FROM companies").fetchdf())

# Check filings
print("\nFilings:")
print(conn.execute("SELECT * FROM filings LIMIT 5").fetchdf())

# Check time series metrics
print("\nTime Series Metrics:")
print(conn.execute("SELECT * FROM time_series_metrics LIMIT 5").fetchdf())

# Check for Revenue metric
print("\nRevenue Metrics:")
print(conn.execute("SELECT * FROM time_series_metrics WHERE metric_name LIKE '%Revenue%' LIMIT 5").fetchdf())

# Check for MSFT Revenue
print("\nMSFT Revenue Metrics:")
print(
    conn.execute(
        "SELECT * FROM time_series_metrics WHERE ticker = 'MSFT' AND metric_name LIKE '%Revenue%' LIMIT 5"
    ).fetchdf()
)

# Close the connection
conn.close()
