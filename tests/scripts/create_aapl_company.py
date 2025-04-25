from sec_filing_analyzer.storage.graph_store import GraphStore

# Initialize Neo4j connection
gs = GraphStore(use_neo4j=True)

# Create AAPL company node
result = gs.query("""
MERGE (c:Company {ticker: 'AAPL'})
SET c.name = 'APPLE INC'
RETURN c.ticker as ticker, c.name as name
""")
print("Created AAPL company node:", result)

# Find AAPL filings
result = gs.query("""
MATCH (f:Filing)
WHERE f.ticker = 'AAPL'
RETURN f.id as filing_id, f.filing_date as filing_date
""")
print("AAPL filings:", result)

# Connect AAPL company to its filings
if result:
    for filing in result:
        filing_id = filing.get("filing_id")
        if filing_id:
            connect_result = gs.query(
                """
            MATCH (c:Company {ticker: 'AAPL'})
            MATCH (f:Filing {id: $filing_id})
            MERGE (c)-[:FILED]->(f)
            RETURN c.ticker as company, f.id as filing
            """,
                filing_id=filing_id,
            )
            print(f"Connected AAPL to filing {filing_id}:", connect_result)

# Verify connections
result = gs.query("""
MATCH (c:Company {ticker: 'AAPL'})-[:FILED]->(f:Filing)
RETURN f.id as filing_id, f.filing_date as filing_date
""")
print("AAPL filings after connection:", result)
