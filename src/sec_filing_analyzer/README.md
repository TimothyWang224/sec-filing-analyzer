# SEC Filing Analyzer

A comprehensive tool for analyzing SEC filings using graph databases and vector embeddings.

## Overview

The SEC Filing Analyzer is a Python package that provides tools for:
- Downloading and processing SEC filings
- Extracting structured data from filings
- Storing filing data in both graph and vector databases
- Analyzing filing content using natural language processing
- Querying filing data using both graph and semantic search

## Installation

```bash
pip install sec-filing-analyzer
```

## Configuration

The package uses environment variables for configuration. Create a `.env` file with:

```env
# Neo4j Configuration
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_URL=bolt://localhost:7687
NEO4J_DATABASE=neo4j

# Storage Configuration
STORAGE_MAX_CLUSTER_SIZE=100
STORAGE_USE_NEO4J=true
STORAGE_VECTOR_STORE_TYPE=llamaindex
STORAGE_VECTOR_STORE_PATH=./data/vector_store

# ETL Configuration
ETL_DEFAULT_FILING_TYPES=["10-K", "10-Q", "8-K"]
ETL_CACHE_DIR=./cache
```

## Usage

### Basic Usage

```python
from sec_filing_analyzer import SECFilingAnalyzer

# Initialize analyzer
analyzer = SECFilingAnalyzer()

# Analyze financials
results = analyzer.analyze_financials(
    ticker="AAPL",
    filing_type="10-K",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Assess risks
risks = analyzer.assess_risks(
    ticker="AAPL",
    filing_type="10-K",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Answer questions
answer = analyzer.answer_question(
    ticker="AAPL",
    question="What are the main risks to the business?",
    filing_type="10-K",
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

### ETL Pipeline

```python
from sec_filing_analyzer.pipeline import SECFilingETLPipeline

# Initialize pipeline
pipeline = SECFilingETLPipeline()

# Process company filings
pipeline.process_company(
    ticker="AAPL",
    filing_types=["10-K", "10-Q"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

## Storage

The package uses a unified storage system that combines:

1. Graph Storage (`GraphStore`)
   - Stores filing structure and relationships
   - Supports both in-memory and Neo4j backends
   - Provides community detection and summarization

2. Vector Storage (`VectorStore`)
   - Stores document embeddings
   - Supports both Pinecone and LlamaIndex backends
   - Enables semantic search

## Development

### Project Structure

```
sec_filing_analyzer/
├── api.py              # Main API interface
├── config.py           # Configuration management
├── storage/            # Storage implementations
│   ├── __init__.py
│   ├── interfaces.py   # Storage interfaces
│   ├── graph_store.py  # Graph storage implementation
│   └── vector_store.py # Vector storage implementation
├── data_retrieval/     # Data retrieval utilities
│   ├── __init__.py
│   ├── filing_processor.py
│   └── sec_downloader.py
├── embeddings/         # Embedding generation
│   ├── __init__.py
│   └── embeddings.py
├── pipeline/           # ETL pipeline
│   ├── __init__.py
│   └── etl_pipeline.py
└── graphrag/          # GraphRAG implementation
    ├── __init__.py
    ├── sec_structure.py
    ├── sec_entities.py
    └── llamaindex_integration.py
```

### Running Tests

```bash
pytest tests/
```

## License

MIT License 