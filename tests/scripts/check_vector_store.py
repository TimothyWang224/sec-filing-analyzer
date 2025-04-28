from sec_filing_analyzer.storage.vector_store import LlamaIndexVectorStore

# Initialize vector store
vector_store = LlamaIndexVectorStore()

# Check for AAPL filings
print("Searching for AAPL filings...")
aapl_filings = vector_store.search_vectors(
    "AAPL", metadata_filter={"ticker": "AAPL"}, top_k=10
)
print(f"Found {len(aapl_filings)} AAPL filings in vector store:")
for filing in aapl_filings:
    print(f"  - ID: {filing['id']}")
    print(f"    Score: {filing['score']}")
    print(f"    Metadata: {filing['metadata']}")
    print(f"    Text length: {len(filing['text'])}")
    print()

# Check for MSFT filings
print("\nSearching for MSFT filings...")
msft_filings = vector_store.search_vectors(
    "MSFT", metadata_filter={"ticker": "MSFT"}, top_k=10
)
print(f"Found {len(msft_filings)} MSFT filings in vector store:")
for filing in msft_filings:
    print(f"  - ID: {filing['id']}")
    print(f"    Score: {filing['score']}")
    print(f"    Metadata: {filing['metadata']}")
    print(f"    Text length: {len(filing['text'])}")
    print()

# Check for NVDA filings
print("\nSearching for NVDA filings...")
nvda_filings = vector_store.search_vectors(
    "NVDA", metadata_filter={"ticker": "NVDA"}, top_k=10
)
print(f"Found {len(nvda_filings)} NVDA filings in vector store:")
for filing in nvda_filings:
    print(f"  - ID: {filing['id']}")
    print(f"    Score: {filing['score']}")
    print(f"    Metadata: {filing['metadata']}")
    print(f"    Text length: {len(filing['text'])}")
    print()
