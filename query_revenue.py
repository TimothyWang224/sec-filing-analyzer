import duckdb

# Connect to the database
conn = duckdb.connect("data/db_backup/financial_data.duckdb", read_only=True)

# Query GOOGL revenue for 2023
query = """
SELECT 
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
    AND financial_facts.metric_name = 'Revenue' 
    AND filings.fiscal_year = 2023
"""

# Execute the query
result = conn.execute(query).fetchdf()
print("GOOGL Revenue for 2023:")
print(result)

# Close the connection
conn.close()
