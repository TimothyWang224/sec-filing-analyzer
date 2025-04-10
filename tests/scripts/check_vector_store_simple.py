from sec_filing_analyzer.storage.vector_store import LlamaIndexVectorStore

# Initialize vector store
vector_store = LlamaIndexVectorStore()

# Check for all filings
print("Searching for all filings...")
all_filings = vector_store.search_vectors("filing", top_k=10)
print(f"Found {len(all_filings)} filings in vector store:")
for filing in all_filings:
    print(f"  - ID: {filing['id']}")
    print(f"    Score: {filing['score']}")
    print(f"    Metadata: {filing['metadata']}")
    print(f"    Text length: {len(filing['text'])}")
    print()
