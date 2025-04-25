from sec_filing_analyzer.quantitative.storage.optimized_duckdb_store import OptimizedDuckDBStore

# Initialize the store
db_store = OptimizedDuckDBStore(db_path="data/db_backup/financial_data.duckdb", read_only=True)

# Test query_financial_facts
print("Testing query_financial_facts for MSFT Revenue in 2022:")
results = db_store.query_financial_facts(
    ticker="MSFT", metrics=["Revenue"], start_date="2022-01-01", end_date="2022-12-31"
)

# Print results
print(f"Found {len(results)} results:")
for result in results:
    print(f"Ticker: {result.get('ticker')}")
    print(f"Metric: {result.get('metric_name')}")
    print(f"Value: {result.get('value')}")
    print(f"Fiscal Year: {result.get('fiscal_year')}")
    print(f"Fiscal Quarter: {result.get('fiscal_quarter')}")
    print()

# Close the connection
db_store.close()
