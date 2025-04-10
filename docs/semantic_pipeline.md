# Semantic ETL Pipeline

This document describes the semantic ETL pipeline for the SEC Filing Analyzer.

## Overview

The semantic ETL pipeline processes SEC filings to extract semantic information, generate embeddings, and store them in a vector database for semantic search. It focuses on the textual content of the filings, enabling natural language queries and semantic similarity search.

## Architecture

The semantic pipeline consists of the following components:

1. **Document Chunking**: Splits documents into semantically meaningful chunks
2. **Embedding Generation**: Generates vector embeddings for each chunk
3. **Vector Storage**: Stores embeddings and metadata for efficient retrieval
4. **Graph Storage**: Stores relationships between filings, sections, and entities

## Components

### Document Chunking

The `DocumentChunker` class in `semantic/processing/chunking.py` handles the chunking of documents:

```python
from sec_filing_analyzer.semantic.processing.chunking import DocumentChunker

# Initialize chunker
chunker = DocumentChunker()

# Chunk a document
chunks = chunker.chunk_document("This is a document to chunk...")
```

The chunker splits documents into chunks of approximately 1500 tokens, which is optimal for retrieval precision. It also preserves metadata about the source document and the position of each chunk within it.

### Embedding Generation

The `EmbeddingGenerator` class in `semantic/embeddings/embedding_generator.py` handles the generation of embeddings:

```python
from sec_filing_analyzer.semantic.embeddings.embedding_generator import EmbeddingGenerator

# Initialize embedding generator
embedding_generator = EmbeddingGenerator()

# Generate embeddings for a text
embedding = embedding_generator.generate_embedding("This is a text to embed...")
```

For parallel processing of multiple chunks, the `ParallelEmbeddingGenerator` class can be used:

```python
from sec_filing_analyzer.semantic.embeddings.parallel_embeddings import ParallelEmbeddingGenerator

# Initialize parallel embedding generator with 4 workers
embedding_generator = ParallelEmbeddingGenerator(num_workers=4)

# Generate embeddings for multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = embedding_generator.generate_embeddings(texts)
```

### Vector Storage

The `VectorStore` class in `semantic/storage/vector_store.py` handles the storage and retrieval of embeddings:

```python
from sec_filing_analyzer.semantic.storage.vector_store import VectorStore

# Initialize vector store
vector_store = VectorStore()

# Add embeddings to the vector store
embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
metadata_list = [{"text": "Text 1"}, {"text": "Text 2"}]
ids = vector_store.add_embeddings(embeddings, metadata_list)

# Search for similar embeddings
query_embedding = [0.1, 0.2, 0.3]
results = vector_store.search(query_embedding, top_k=5)
```

## Pipeline Flow

The semantic ETL pipeline follows these steps:

1. **Download Filing**: Download the SEC filing using the SEC downloader
2. **Chunk Document**: Split the document into semantically meaningful chunks
3. **Generate Embeddings**: Generate embeddings for each chunk
4. **Store Embeddings**: Store the embeddings and metadata in the vector store
5. **Build Graph**: Build a graph of relationships between filings, sections, and entities

## Usage

```python
from sec_filing_analyzer.pipeline import SemanticETLPipeline

# Initialize semantic pipeline
pipeline = SemanticETLPipeline()

# Process a filing
result = pipeline.process_filing(
    ticker="AAPL",
    filing_type="10-K",
    filing_date="2023-01-01"
)

# Check the result
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Successfully processed filing: {result}")
```

## Configuration

The semantic pipeline can be configured with the following parameters:

- `chunk_size`: Size of document chunks (default: 1500 tokens)
- `chunk_overlap`: Overlap between chunks (default: 150 tokens)
- `embedding_model`: OpenAI embedding model to use (default: "text-embedding-3-small")
- `use_parallel`: Whether to use parallel processing (default: True)
- `num_workers`: Number of workers for parallel processing (default: 4)

Example:

```python
from sec_filing_analyzer.pipeline import SemanticETLPipeline

# Initialize semantic pipeline with custom configuration
pipeline = SemanticETLPipeline(
    chunk_size=2000,
    chunk_overlap=200,
    embedding_model="text-embedding-3-large",
    use_parallel=True,
    num_workers=8
)
```

## Performance Considerations

- **Chunk Size**: Smaller chunks provide better retrieval precision but require more embeddings
- **Parallel Processing**: Use parallel processing for large documents or many filings
- **Embedding Model**: Larger models provide better quality but are more expensive and slower
- **Memory Usage**: Loading all embeddings into memory can be memory-intensive for large datasets
