import duckdb

# Connect to the database
conn = duckdb.connect("data/db_backup/improved_financial_data.duckdb", read_only=True)

# Check metrics table schema
print("Schema of metrics table:")
metrics_schema = conn.execute("DESCRIBE metrics").fetchdf()
print(metrics_schema)

# Check facts table schema
print("\nSchema of facts table:")
facts_schema = conn.execute("DESCRIBE facts").fetchdf()
print(facts_schema)

# Check sample data from metrics
print("\nSample data from metrics table:")
metrics_sample = conn.execute("SELECT * FROM metrics LIMIT 5").fetchdf()
print(metrics_sample)

# Check sample data from facts
print("\nSample data from facts table:")
facts_sample = conn.execute("SELECT * FROM facts LIMIT 5").fetchdf()
print(facts_sample)

# Check sample data from time_series_view
print("\nSample data from time_series_view:")
time_series_sample = conn.execute("SELECT * FROM time_series_view LIMIT 5").fetchdf()
print(time_series_sample)

# Check if there's any data in time_series_view
print("\nCount of records in time_series_view:")
count = conn.execute("SELECT COUNT(*) FROM time_series_view").fetchone()[0]
print(f"Total records: {count}")

# Close the connection
conn.close()
