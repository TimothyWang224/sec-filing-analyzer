# SEC Filing Analyzer

A comprehensive tool for analyzing SEC filings using graph databases and vector embeddings.

## Features

- **ETL Pipeline**: Process SEC filings (10-K, 10-Q, 8-K) with structured data extraction
- **Graph Database**: Store filing data in Neo4j with rich relationships
- **Vector Embeddings**: Generate and store embeddings using OpenAI's API
- **Vector Search**: Efficient similarity search using LlamaIndex
- **Entity Extraction**: Identify companies, people, and financial concepts
- **Topic Analysis**: Extract topics from filing sections
- **Intelligent Chunking**: Semantic chunking of documents with a 1500 token size for optimal retrieval

## Architecture

The system uses a dual-storage approach:

1. **Neo4j Graph Database**:
   - Stores structured filing data
   - Maintains relationships between filings, sections, and entities
   - References vector IDs for semantic search

2. **LlamaIndex Vector Store**:
   - Stores vector embeddings for semantic search
   - Provides efficient similarity search with cosine similarity
   - Maintains metadata for filtering and retrieval

## Setup

1. Install dependencies:
   ```bash
   # Install using poetry
   poetry install

   # Or install dependencies directly
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`:
   ```
   EDGAR_IDENTITY=your_edgar_identity
   OPENAI_API_KEY=your_openai_api_key
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

3. Initialize Neo4j database:
   ```bash
   python -m sec_filing_analyzer.init_db
   ```

## Usage

### Process Filings

```python
from sec_filing_analyzer.pipeline import SECFilingETLPipeline

# Initialize pipeline
pipeline = SECFilingETLPipeline()

# Process filings for a company
pipeline.process_company(
    ticker="AAPL",
    filing_types=["10-K", "10-Q"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### Search Similar Content

```python
from sec_filing_analyzer.storage import LlamaIndexVectorStore

# Initialize vector store
vector_store = LlamaIndexVectorStore()

# Search for similar content
results = vector_store.search_vectors(
    query_vector="revenue concentration",  # Use a text query instead of embedding
    top_k=5
)

# Display results
for result in results:
    print(f"Document ID: {result['id']}")
    print(f"Score: {result['score']}")
    print(f"Text: {result['text'][:100]}...\n")
```

## Project Structure

```
sec_filing_analyzer/
├── config.py             # Configuration management
├── data_processing/      # Document processing and chunking
│   └── chunking.py       # Intelligent document chunking (1500 tokens)
├── data_retrieval/       # SEC filing retrieval
│   ├── file_storage.py   # Local file storage
│   ├── filing_processor.py # Filing processing
│   └── sec_downloader.py # SEC EDGAR downloader
├── embeddings/           # Embedding generation
│   └── embeddings.py     # OpenAI embedding generation
├── graphrag/             # Graph RAG components
├── pipeline/             # ETL pipeline
│   └── etl_pipeline.py   # Main ETL pipeline
├── storage/              # Storage implementations
│   ├── graph_store.py    # Graph database interface
│   ├── interfaces.py     # Storage interfaces
│   └── vector_store.py   # Vector storage and search
└── tests/                # Test scripts
```

## Document Processing

### Intelligent Chunking

The system uses intelligent chunking to process SEC filings:

- **Chunk Size**: 1500 tokens per chunk for optimal retrieval precision
- **Semantic Chunking**: Documents are chunked based on semantic boundaries
- **Sub-chunking**: Large chunks are automatically split into smaller sub-chunks
- **Metadata Preservation**: Each chunk maintains metadata about its source filing

### Vector Embeddings

- **OpenAI Embeddings**: Uses OpenAI's embedding models for high-quality vector representations
- **Cosine Similarity**: Search uses cosine similarity for accurate retrieval
- **Metadata Filtering**: Results can be filtered by ticker, filing type, year, etc.

## Dependencies

- `edgar`: SEC filing retrieval and parsing
- `neo4j`: Graph database
- `llama-index`: Vector storage and search
- `openai`: Embedding generation
- `tiktoken`: Token counting for chunking
- `rich`: Terminal UI
- `numpy`: Numerical operations for embeddings
- `pandas`: Data manipulation for filing data

## Scripts

- `scripts/run_nvda_etl.py`: Run the ETL pipeline for NVIDIA
- `scripts/explore_vector_store.py`: Search for similar content in filings
- `scripts/direct_search.py`: Direct search using cosine similarity
- `scripts/analyze_topics.py`: Extract topics from filings
- `scripts/visualize_graph.py`: Visualize the graph database

## License

MIT
