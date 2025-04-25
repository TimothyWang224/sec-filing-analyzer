from sec_filing_analyzer.storage.graph_store import GraphStore

# Initialize Neo4j connection
gs = GraphStore(use_neo4j=True)

try:
    # Check for all companies
    result = gs.query("MATCH (c) WHERE c.ticker IS NOT NULL RETURN c.ticker as ticker, labels(c) as labels")
    print("All nodes with ticker property:", result)

    # Check for AAPL specifically
    result = gs.query("MATCH (c) WHERE c.ticker = 'AAPL' RETURN c.ticker as ticker, labels(c) as labels")
    print("AAPL nodes:", result)

    # Check for MSFT specifically
    result = gs.query("MATCH (c) WHERE c.ticker = 'MSFT' RETURN c.ticker as ticker, labels(c) as labels")
    print("MSFT nodes:", result)
finally:
    # Ensure the Neo4j driver is closed
    gs.close()

# Check for filings
result = gs.query("MATCH (f:Filing) RETURN f.ticker as ticker, count(f) as filing_count")
print("Filings by ticker:", result)

# Check for chunks
result = gs.query("MATCH (c:Chunk)-[:CONTAINS]-(f:Filing) WHERE f.ticker = 'AAPL' RETURN count(c) as chunk_count")
print("AAPL chunk count:", result)
