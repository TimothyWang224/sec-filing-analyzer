import json

# Load the cached filing
with open('data/filings/cache/0001045810-25-000082.json', 'r') as f:
    data = json.load(f)

# Check the chunk embeddings
chunk_embeddings = data['processed_data']['chunk_embeddings']
print(f"Type of chunk_embeddings: {type(chunk_embeddings)}")
print(f"Length of chunk_embeddings: {len(chunk_embeddings)}")
print(f"Type of first chunk embedding: {type(chunk_embeddings[0])}")
print(f"Length of first chunk embedding: {len(chunk_embeddings[0])}")

# Print the first few values of the first chunk embedding
print(f"First few values of first chunk embedding: {chunk_embeddings[0][:5]}")

# Check the embedding metadata
embedding_metadata = data['processed_data']['embedding_metadata']
print(f"Type of embedding_metadata: {type(embedding_metadata)}")
print(f"Keys in embedding_metadata: {embedding_metadata.keys()}")

# Check if the chunk embeddings are being used correctly
print("\nChecking if chunk embeddings are being used correctly...")
for i, chunk_embedding in enumerate(chunk_embeddings[:2]):
    print(f"Chunk {i}:")
    print(f"  Type: {type(chunk_embedding)}")
    print(f"  Length: {len(chunk_embedding)}")
    print(f"  First few values: {chunk_embedding[:5]}")
