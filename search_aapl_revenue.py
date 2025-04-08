from sec_filing_analyzer.storage.vector_store import LlamaIndexVectorStore

# Initialize vector store
vector_store = LlamaIndexVectorStore()

# Search for AAPL revenue information
print("Searching for AAPL revenue information...")
aapl_revenue = vector_store.search_vectors("AAPL revenue and financial performance", top_k=5)
print(f"Found {len(aapl_revenue)} results:")
for result in aapl_revenue:
    print(f"  - ID: {result['id']}")
    print(f"    Score: {result['score']}")
    print(f"    Metadata: {result['metadata']}")
    print(f"    Text: {result['text'][:500]}...")
    print()
