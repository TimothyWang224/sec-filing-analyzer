# Codebase Reorganization

This document describes the reorganization of the SEC Filing Analyzer codebase to separate semantic and quantitative data processing.

## Motivation

The original codebase had all data processing logic in a single pipeline, which made it difficult to maintain and extend. The reorganization separates semantic and quantitative data processing into separate modules, making the code more modular and easier to maintain.

## Directory Structure Changes

### Before

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

### After

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

## Pipeline Changes

### Before

The original codebase had a single ETL pipeline that handled both semantic and quantitative data processing:

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

### After

The reorganized codebase has separate pipelines for semantic and quantitative data processing, as well as a unified pipeline that can use both:

#### Unified Pipeline

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

#### Semantic Pipeline Only

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

#### Quantitative Pipeline Only

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

## Benefits of the Reorganization

1. **Modularity**: Each module can be developed, tested, and maintained independently.
2. **Flexibility**: Users can choose to process only semantic data, only quantitative data, or both.
3. **Scalability**: Each pipeline can be scaled independently based on its specific requirements.
4. **Reusability**: Components can be reused in different contexts.
5. **Maintainability**: The code is easier to understand and maintain.

## Testing the Reorganization

Two test scripts have been created to verify that the reorganized codebase works correctly:

1. `test_reorganized_pipeline.py`: Tests the reorganized ETL pipeline
2. `test_reorganized_structure.py`: Tests the reorganized directory structure

To run the tests:

```bash
# Test the reorganized directory structure
python src/scripts/test_reorganized_structure.py

# Test the semantic pipeline
python src/scripts/test_reorganized_pipeline.py --mode=semantic --ticker=MSFT --filing-type=10-K

# Test the quantitative pipeline
python src/scripts/test_reorganized_pipeline.py --mode=quantitative --ticker=MSFT --filing-type=10-K

# Test the unified pipeline
python src/scripts/test_reorganized_pipeline.py --mode=unified --ticker=MSFT --filing-type=10-K
```
