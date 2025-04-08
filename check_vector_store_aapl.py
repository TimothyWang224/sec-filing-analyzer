from sec_filing_analyzer.storage.vector_store import LlamaIndexVectorStore

# Initialize vector store
vector_store = LlamaIndexVectorStore()

# Check for AAPL filings
print("Searching for AAPL filings...")
aapl_filings = vector_store.search_vectors("AAPL", top_k=10)
print(f"Found {len(aapl_filings)} AAPL filings in vector store:")
for filing in aapl_filings:
    print(f"  - ID: {filing['id']}")
    print(f"    Score: {filing['score']}")
    print(f"    Metadata: {filing['metadata']}")
    print(f"    Text length: {len(filing['text'])}")
    print()
