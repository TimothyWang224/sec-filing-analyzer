# ETL Pipeline Organization

This document describes the organization of the ETL pipeline for processing SEC filings, including both semantic (text) and quantitative (XBRL) data.

## Overview

The SEC Filing Analyzer ETL pipeline is designed to extract, transform, and load both semantic and quantitative data from SEC filings. The pipeline is organized into separate modules for each type of data, with a unified interface for processing filings.

## Directory Structure

```
src/sec_filing_analyzer/
├── data_retrieval/           # Common retrieval code for all data types
│   ├── sec_downloader.py     # Downloads SEC filings
│   ├── file_storage.py       # Stores filing files
│   └── ...
├── semantic/                 # Semantic (text) data processing
│   ├── processing/           # Text processing
│   │   ├── chunking.py       # Chunks documents into smaller pieces
│   │   └── ...
│   ├── storage/              # Vector storage
│   │   ├── vector_store.py   # Stores embeddings
│   │   └── ...
│   └── embeddings/           # Embedding generation
│       └── ...
├── quantitative/             # Quantitative (XBRL) data processing
│   ├── processing/           # XBRL processing
│   │   ├── xbrl_extractor.py # Extracts XBRL data
│   │   └── ...
│   └── storage/              # Relational storage
│       ├── duckdb_store.py   # Stores financial data in DuckDB
│       └── ...
├── pipeline/                 # ETL pipeline orchestration
│   ├── etl_pipeline.py       # Main ETL pipeline
│   ├── semantic_pipeline.py  # Semantic-specific pipeline
│   ├── quantitative_pipeline.py  # Quantitative-specific pipeline
│   └── ...
└── utils/                    # Common utilities
    └── ...
```

## Pipeline Components

### 1. Data Retrieval

The `data_retrieval` module is responsible for downloading SEC filings and storing them locally. It provides a common interface for both semantic and quantitative data processing.

Key components:
- `SECFilingsDownloader`: Downloads SEC filings from the SEC EDGAR database
- `FileStorage`: Stores filing files locally

### 2. Semantic Data Processing

The `semantic` module is responsible for processing and storing semantic (text) data from SEC filings.

Key components:
- `processing/chunking.py`: Chunks documents into smaller pieces for embedding
- `embeddings/`: Generates embeddings for document chunks
- `storage/vector_store.py`: Stores embeddings in a vector database

### 3. Quantitative Data Processing

The `quantitative` module is responsible for processing and storing quantitative (XBRL) data from SEC filings.

Key components:
- `processing/xbrl_extractor.py`: Extracts XBRL data from SEC filings
- `processing/edgar_xbrl_to_duckdb.py`: Extracts XBRL data using the edgar library and stores it in DuckDB
- `storage/duckdb_store.py`: Stores financial data in a DuckDB database

### 4. Pipeline Orchestration

The `pipeline` module is responsible for orchestrating the ETL process.

Key components:
- `etl_pipeline.py`: Main ETL pipeline that coordinates both semantic and quantitative processing
- `semantic_pipeline.py`: Semantic-specific pipeline
- `quantitative_pipeline.py`: Quantitative-specific pipeline

## Usage

### Processing a Single Filing

```python
from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline

# Initialize the ETL pipeline
pipeline = SECFilingETLPipeline(
    process_semantic=True,     # Process semantic data
    process_quantitative=True, # Process quantitative data
    db_path="data/financial_data.duckdb"
)

# Process a single filing
result = pipeline.process_filing(
    ticker="MSFT",
    filing_type="10-K",
    filing_date="2022-07-28"
)

print(result)
```

### Processing Multiple Filings for a Company

```python
# Process multiple filings for a company
result = pipeline.process_company(
    ticker="MSFT",
    filing_types=["10-K", "10-Q"],
    start_date="2020-01-01",
    end_date="2022-12-31"
)

print(result)
```

## Benefits of Separation

The separation of semantic and quantitative data processing provides several benefits:

1. **Modularity**: Each module can be developed, tested, and maintained independently.
2. **Flexibility**: Users can choose to process only semantic data, only quantitative data, or both.
3. **Scalability**: Each pipeline can be scaled independently based on its specific requirements.
4. **Clarity**: The code is more organized and easier to understand.
5. **Reusability**: Components can be reused in different contexts.

## Conclusion

The reorganized ETL pipeline provides a clear separation of concerns between semantic and quantitative data processing, while maintaining a unified interface for processing SEC filings. This organization makes the codebase more maintainable, flexible, and scalable.
