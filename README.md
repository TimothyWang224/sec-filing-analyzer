# SEC Filing Analyzer

A comprehensive tool for analyzing SEC filings using graph databases and vector embeddings.

## Features

- **ETL Pipeline**: Process SEC filings (10-K, 10-Q, 8-K) with structured data extraction
- **Graph Database**: Store filing data in Neo4j with rich relationships
- **Vector Embeddings**: Generate and store embeddings using OpenAI's API
- **Vector Search**: Efficient similarity search using LlamaIndex
- **Entity Extraction**: Identify companies, people, and financial concepts
- **Topic Analysis**: Extract topics from filing sections

## Architecture

The system uses a dual-storage approach:

1. **Neo4j Graph Database**:
   - Stores structured filing data
   - Maintains relationships between filings, sections, and entities
   - References vector IDs for semantic search

2. **LlamaIndex Vector Store**:
   - Stores vector embeddings for semantic search
   - Provides efficient similarity search
   - Maintains metadata for filtering

## Setup

1. Install dependencies:
   ```bash
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
pipeline = SECFilingETLPipeline(
    output_dir="output/sec_filings",
    cache_dir="cache",
    use_neo4j=True,
    use_vector_store=True
)

# Process filings for a company
results = pipeline.process_company(
    ticker="AAPL",
    years=[2023],
    filing_types=["10-K", "10-Q"]
)
```

### Search Similar Content

```python
from sec_filing_analyzer.vector_store import LlamaIndexVectorStore

# Initialize vector store
vector_store = LlamaIndexVectorStore()

# Search for similar content
results = vector_store.search_vectors(
    query_vector=embedding,
    top_k=10,
    filter_metadata={"ticker": "AAPL", "year": 2023}
)
```

## Project Structure

```
sec_filing_analyzer/
├── embeddings/           # Embedding generation
├── graph_store/          # Neo4j graph database interface
├── graphrag/             # Graph RAG components
├── pipeline/             # ETL pipeline
├── vector_store/         # Vector storage and search
└── tests/                # Test scripts
```

## Dependencies

- `edgartools`: SEC filing retrieval and parsing
- `neo4j`: Graph database
- `llama-index`: Vector storage and search
- `openai`: Embedding generation
- `rich`: Terminal UI

## License

MIT
