import duckdb

# Connect to the database
conn = duckdb.connect('data/db_backup/financial_data.duckdb', read_only=True)

# Check what companies are available
companies_query = """
SELECT * FROM companies
"""
companies_result = conn.execute(companies_query).fetchdf()
print("Companies in the database:")
print(companies_result)

# Check what filings are available
filings_query = """
SELECT ticker, COUNT(*) as filing_count FROM filings GROUP BY ticker
"""
filings_result = conn.execute(filings_query).fetchdf()
print("\nFilings by company:")
print(filings_result)

# Close the connection
conn.close()
