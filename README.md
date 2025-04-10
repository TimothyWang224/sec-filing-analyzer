# SEC Filing Analyzer

A comprehensive tool for analyzing SEC filings using graph databases and vector embeddings.

## Features

### Semantic Data Processing

- **Intelligent Chunking**: Semantic chunking of documents with a 1500 token size for optimal retrieval
- **Vector Embeddings**: Generate and store embeddings using OpenAI's API
- **Vector Search**: Efficient similarity search using LlamaIndex
- **Graph Database**: Store filing data in Neo4j with rich relationships
- **Entity Extraction**: Identify companies, people, and financial concepts
- **Topic Analysis**: Extract topics from filing sections

### Quantitative Data Processing

- **XBRL Extraction**: Extract structured financial data from XBRL filings
- **DuckDB Storage**: Store financial data in DuckDB for efficient querying
- **Financial Metrics**: Extract key financial metrics from SEC filings
- **Time-Series Analysis**: Support for time-series analysis of financial data

### Unified ETL Pipeline

- **Modular Architecture**: Process semantic and quantitative data separately or together
- **Parallel Processing**: Process multiple filings in parallel for improved performance
- **Incremental Updates**: Support for incremental updates to the database

## Architecture

The system uses a modular architecture with separate pipelines for semantic and quantitative data processing:

### Semantic Data Processing

1. **Neo4j Graph Database**:
   - Stores structured filing data
   - Maintains relationships between filings, sections, and entities
   - References vector IDs for semantic search

2. **LlamaIndex Vector Store**:
   - Stores vector embeddings for semantic search
   - Provides efficient similarity search with cosine similarity
   - Maintains metadata for filtering and retrieval

### Quantitative Data Processing

1. **DuckDB Database**:
   - Stores structured financial data extracted from XBRL
   - Provides efficient SQL queries for financial analysis
   - Supports time-series analysis of financial metrics

### Unified ETL Pipeline

The system provides a unified ETL pipeline that can process both semantic and quantitative data, or either one separately, depending on the user's needs.

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

### Unified Pipeline

```python
from sec_filing_analyzer.pipeline import SECFilingETLPipeline

# Initialize pipeline with both semantic and quantitative processing
pipeline = SECFilingETLPipeline(process_semantic=True, process_quantitative=True)

# Process filings for a company
pipeline.process_company(
    ticker="AAPL",
    filing_types=["10-K", "10-Q"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### Semantic Pipeline Only

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
```

### Quantitative Pipeline Only

```python
from sec_filing_analyzer.pipeline import QuantitativeETLPipeline

# Initialize quantitative pipeline
pipeline = QuantitativeETLPipeline(db_path="data/financial_data.duckdb")

# Process a filing
result = pipeline.process_filing(
    ticker="AAPL",
    filing_type="10-K",
    filing_date="2023-01-01"
)
```

### Search Similar Content

```python
from sec_filing_analyzer.semantic.storage import VectorStore

# Initialize vector store
vector_store = VectorStore()

# Search for similar content
results = vector_store.search(
    query_embedding=vector_store.embedding_generator.generate_embedding("revenue concentration"),
    top_k=5
)

# Display results
for result in results:
    print(f"Document ID: {result['id']}")
    print(f"Score: {result['score']}")
    print(f"Text: {result['metadata'].get('text', '')[:100]}...\n")
```

### Query Financial Data

```python
from sec_filing_analyzer.quantitative.storage import OptimizedDuckDBStore

# Initialize DuckDB store
db_store = OptimizedDuckDBStore(db_path="data/financial_data.duckdb")

# Query financial data
results = db_store.query_financial_facts(
    ticker="AAPL",
    metrics=["Revenue", "NetIncome"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# Display results
for result in results:
    print(f"Ticker: {result['ticker']}")
    print(f"Metric: {result['metric_name']}")
    print(f"Value: {result['value']}")
    print(f"Period: {result['period_end_date']}\n")
```

## Project Structure

```
sec_filing_analyzer/
├── config.py                # Configuration management
├── data_retrieval/          # SEC filing retrieval
│   ├── file_storage.py      # Local file storage
│   ├── filing_processor.py  # Filing processing
│   └── sec_downloader.py    # SEC EDGAR downloader
├── graphrag/                # Graph RAG components
├── pipeline/                # ETL pipeline
│   ├── etl_pipeline.py      # Main ETL pipeline
│   ├── semantic_pipeline.py # Semantic data processing pipeline
│   └── quantitative_pipeline.py # Quantitative data processing pipeline
├── semantic/                # Semantic data processing
│   ├── processing/          # Document processing
│   │   └── chunking.py      # Intelligent document chunking (1500 tokens)
│   ├── embeddings/          # Embedding generation
│   │   ├── embedding_generator.py # OpenAI embedding generation
│   │   └── parallel_embeddings.py # Parallel embedding generation
│   └── storage/             # Semantic data storage
│       └── vector_store.py  # Vector storage and search
├── quantitative/            # Quantitative data processing
│   ├── processing/          # XBRL data extraction
│   │   └── edgar_xbrl_to_duckdb.py # XBRL to DuckDB extraction
│   └── storage/             # Quantitative data storage
│       └── optimized_duckdb_store.py # DuckDB storage
├── storage/                 # Common storage implementations
│   ├── graph_store.py       # Graph database interface
│   └── interfaces.py        # Storage interfaces
└── tests/                   # Test scripts
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

### Core Dependencies

- `edgar`: SEC filing retrieval and parsing (aliased as `edgartools`)
- `openai`: API access for embedding generation
- `tiktoken`: Token counting for chunking
- `rich`: Terminal UI
- `numpy`: Numerical operations for embeddings
- `pandas`: Data manipulation for filing data

### Semantic Processing

- `neo4j`: Graph database for storing relationships
- `llama-index-core`: Vector storage and search
- `llama-index-embeddings-openai`: OpenAI embeddings integration
- `faiss-cpu`: Efficient vector similarity search

### Quantitative Processing

- `duckdb`: Efficient analytical database for financial data
- `pyarrow`: Efficient data interchange format

## Scripts

### ETL Scripts

- `scripts/run_nvda_etl.py`: Run the ETL pipeline for NVIDIA
- `scripts/test_reorganized_pipeline.py`: Test the reorganized ETL pipeline
- `scripts/test_reorganized_structure.py`: Test the reorganized directory structure

### Analysis Scripts

- `scripts/explore_vector_store.py`: Search for similar content in filings
- `scripts/direct_search.py`: Direct search using cosine similarity
- `scripts/analyze_topics.py`: Extract topics from filings
- `scripts/visualize_graph.py`: Visualize the graph database
- `scripts/query_financial_data.py`: Query financial data from DuckDB

## License

MIT
