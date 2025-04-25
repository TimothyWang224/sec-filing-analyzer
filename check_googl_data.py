import duckdb

# Connect to the database
conn = duckdb.connect("data/db_backup/financial_data.duckdb", read_only=True)

# Check what filings are available for GOOGL
filings_query = """
SELECT * FROM filings WHERE ticker = 'GOOGL'
"""
filings_result = conn.execute(filings_query).fetchdf()
print("GOOGL Filings:")
print(filings_result)

# Check what financial facts are available for GOOGL
facts_query = """
SELECT 
    financial_facts.metric_name,
    financial_facts.value,
    financial_facts.end_date,
    filings.fiscal_year,
    filings.fiscal_quarter
FROM 
    financial_facts 
JOIN 
    filings ON financial_facts.filing_id = filings.id 
WHERE 
    filings.ticker = 'GOOGL'
"""
facts_result = conn.execute(facts_query).fetchdf()
print("\nGOOGL Financial Facts:")
print(facts_result)

# Close the connection
conn.close()
