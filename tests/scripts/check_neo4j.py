from sec_filing_analyzer.storage.graph_store import GraphStore

# Initialize Neo4j connection
gs = GraphStore(use_neo4j=True)

try:
    # Check for all companies
    result = gs.query("MATCH (c:Company) RETURN c.ticker as ticker")
    print("All companies:", result)

    # Check for AAPL specifically
    result = gs.query("MATCH (c:Company) WHERE c.ticker = 'AAPL' RETURN c.ticker as ticker")
    print("AAPL company:", result)

    # Check for MSFT specifically
    result = gs.query("MATCH (c:Company) WHERE c.ticker = 'MSFT' RETURN c.ticker as ticker")
    print("MSFT company:", result)

    # Check for filings
    result = gs.query("MATCH (f:Filing) RETURN f.filing_type as filing_type, f.ticker as ticker LIMIT 5")
    print("Filings:", result)
finally:
    # Ensure the Neo4j driver is closed
    gs.close()

# Check for chunks
result = gs.query("MATCH (c:Chunk) RETURN count(c) as chunk_count")
print("Chunk count:", result)
