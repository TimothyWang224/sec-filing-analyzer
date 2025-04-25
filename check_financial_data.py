import duckdb

# Connect to the database
conn = duckdb.connect("data/db_backup/financial_data.duckdb", read_only=True)

# Check tables
print("Tables in database:")
tables = conn.execute("SHOW TABLES").fetchdf()
print(tables)

# Check companies
print("\nCompanies in database:")
companies = conn.execute("SELECT * FROM companies").fetchdf()
print(companies)

# Check filings
print("\nSample filings (up to 5):")
filings = conn.execute("SELECT * FROM filings LIMIT 5").fetchdf()
print(filings)

# Count filings by ticker
print("\nFilings count by ticker:")
filings_count = conn.execute("""
    SELECT ticker, COUNT(*) as count 
    FROM filings 
    GROUP BY ticker
""").fetchdf()
print(filings_count)

# Check time_series_metrics
print("\nSample time_series_metrics (up to 5):")
metrics = conn.execute("SELECT * FROM time_series_metrics LIMIT 5").fetchdf()
print(metrics)

# Count metrics by ticker
print("\nMetrics count by ticker:")
metrics_count = conn.execute("""
    SELECT ticker, COUNT(*) as count 
    FROM time_series_metrics 
    GROUP BY ticker
""").fetchdf()
print(metrics_count)

# Check revenue data for each company
tickers = ["NVDA", "GOOGL", "AAPL", "MSFT"]
for ticker in tickers:
    print(f"\nRevenue data for {ticker}:")
    revenue = conn.execute(f"""
        SELECT fiscal_year, fiscal_quarter, value 
        FROM time_series_metrics 
        WHERE ticker = '{ticker}' AND metric_name = 'Revenue'
        ORDER BY fiscal_year DESC, fiscal_quarter DESC
    """).fetchdf()
    print(revenue)

# Close the connection
conn.close()
