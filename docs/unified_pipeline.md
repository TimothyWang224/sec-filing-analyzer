# Unified ETL Pipeline

This document describes the unified ETL pipeline for the SEC Filing Analyzer.

## Overview

The unified ETL pipeline combines the semantic and quantitative pipelines into a single pipeline that can process both semantic and quantitative data from SEC filings. It provides a flexible interface that allows users to choose which types of data to process.

## Architecture

The unified pipeline consists of the following components:

1. **Semantic Pipeline**: Processes semantic data from SEC filings
2. **Quantitative Pipeline**: Processes quantitative data from SEC filings
3. **Unified Interface**: Provides a unified interface for both pipelines

## Components

### Semantic Pipeline

The semantic pipeline processes textual content from SEC filings:

- **Document Chunking**: Splits documents into semantically meaningful chunks
- **Embedding Generation**: Generates vector embeddings for each chunk
- **Vector Storage**: Stores embeddings and metadata for efficient retrieval
- **Graph Storage**: Stores relationships between filings, sections, and entities

For more details, see the [Semantic Pipeline documentation](semantic_pipeline.md).

### Quantitative Pipeline

The quantitative pipeline processes structured financial data from SEC filings:

- **XBRL Extraction**: Extracts structured financial data from XBRL filings
- **DuckDB Storage**: Stores financial data in DuckDB for efficient querying
- **Financial Metrics**: Extracts key financial metrics from SEC filings
- **Time-Series Analysis**: Supports time-series analysis of financial data

For more details, see the [Quantitative Pipeline documentation](quantitative_pipeline.md).

### Unified Interface

The unified pipeline provides a single interface for both semantic and quantitative processing:

```python
from sec_filing_analyzer.pipeline import SECFilingETLPipeline

# Initialize pipeline with both semantic and quantitative processing
pipeline = SECFilingETLPipeline(
    process_semantic=True,
    process_quantitative=True,
    db_path="data/financial_data.duckdb"
)

# Process filings for a company
pipeline.process_company(
    ticker="AAPL",
    filing_types=["10-K", "10-Q"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

## Pipeline Flow

The unified ETL pipeline follows these steps:

1. **Download Filing**: Download the SEC filing using the SEC downloader
2. **Process Semantic Data** (if enabled):
   - Chunk the document
   - Generate embeddings
   - Store embeddings and metadata
   - Build graph relationships
3. **Process Quantitative Data** (if enabled):
   - Extract XBRL data
   - Store company and filing information
   - Store financial facts
   - Generate time series metrics
4. **Return Results**: Return results from both pipelines

## Usage

### Process a Single Filing

```python
from sec_filing_analyzer.pipeline import SECFilingETLPipeline

# Initialize pipeline
pipeline = SECFilingETLPipeline(
    process_semantic=True,
    process_quantitative=True
)

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

### Process Multiple Filings for a Company

```python
from sec_filing_analyzer.pipeline import SECFilingETLPipeline

# Initialize pipeline
pipeline = SECFilingETLPipeline(
    process_semantic=True,
    process_quantitative=True
)

# Process filings for a company
result = pipeline.process_company(
    ticker="AAPL",
    filing_types=["10-K", "10-Q"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Check the result
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Successfully processed {result['filings_processed']} filings")
```

### Process Multiple Companies

```python
from sec_filing_analyzer.pipeline import SECFilingETLPipeline

# Initialize pipeline
pipeline = SECFilingETLPipeline(
    process_semantic=True,
    process_quantitative=True
)

# Process filings for multiple companies
companies = ["AAPL", "MSFT", "GOOGL"]
results = {}

for ticker in companies:
    result = pipeline.process_company(
        ticker=ticker,
        filing_types=["10-K", "10-Q"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    results[ticker] = result

# Check the results
for ticker, result in results.items():
    if "error" in result:
        print(f"Error processing {ticker}: {result['error']}")
    else:
        print(f"Successfully processed {result['filings_processed']} filings for {ticker}")
```

## Configuration

The unified pipeline can be configured with the following parameters:

- `process_semantic`: Whether to process semantic data (default: True)
- `process_quantitative`: Whether to process quantitative data (default: True)
- `db_path`: Path to the DuckDB database file (default: "data/financial_data.duckdb")
- `use_parallel`: Whether to use parallel processing (default: True)
- `num_workers`: Number of workers for parallel processing (default: 4)

Example:

```python
from sec_filing_analyzer.pipeline import SECFilingETLPipeline

# Initialize pipeline with custom configuration
pipeline = SECFilingETLPipeline(
    process_semantic=True,
    process_quantitative=True,
    db_path="data/custom_financial_data.duckdb",
    use_parallel=True,
    num_workers=8
)
```

## Performance Considerations

- **Selective Processing**: Process only the data you need (semantic or quantitative)
- **Parallel Processing**: Use parallel processing for improved performance
- **Batch Processing**: Process filings in batches for improved performance
- **Memory Usage**: Monitor memory usage when processing large datasets
