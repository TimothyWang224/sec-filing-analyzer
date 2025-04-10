# Quantitative ETL Pipeline

This document describes the quantitative ETL pipeline for the SEC Filing Analyzer.

## Overview

The quantitative ETL pipeline processes SEC filings to extract structured financial data from XBRL (eXtensible Business Reporting Language) and store it in a DuckDB database for efficient querying and analysis. It focuses on the numerical and financial aspects of the filings, enabling financial analysis and time-series queries.

## Architecture

The quantitative pipeline consists of the following components:

1. **XBRL Extraction**: Extracts structured financial data from XBRL filings
2. **DuckDB Storage**: Stores financial data in DuckDB for efficient querying
3. **Financial Metrics**: Extracts key financial metrics from SEC filings
4. **Time-Series Analysis**: Supports time-series analysis of financial data

## Components

### XBRL Extraction

The `EdgarXBRLToDuckDBExtractor` class in `quantitative/processing/edgar_xbrl_to_duckdb.py` handles the extraction of XBRL data:

```python
from sec_filing_analyzer.quantitative.processing.edgar_xbrl_to_duckdb import EdgarXBRLToDuckDBExtractor

# Initialize XBRL extractor
xbrl_extractor = EdgarXBRLToDuckDBExtractor(db_path="data/financial_data.duckdb")

# Process a filing
result = xbrl_extractor.process_filing(
    ticker="AAPL",
    accession_number="0000320193-23-000077"
)
```

The XBRL extractor uses the edgar library to parse XBRL data and extract financial facts, which are then stored in the DuckDB database.

### DuckDB Storage

The `OptimizedDuckDBStore` class in `quantitative/storage/optimized_duckdb_store.py` handles the storage and retrieval of financial data:

```python
from sec_filing_analyzer.quantitative.storage.optimized_duckdb_store import OptimizedDuckDBStore

# Initialize DuckDB store
db_store = OptimizedDuckDBStore(db_path="data/financial_data.duckdb")

# Query financial facts
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
    print(f"Period: {result['period_end_date']}")
```

## Database Schema

The DuckDB database uses the following schema:

### Companies Table

Stores information about companies:

```sql
CREATE TABLE IF NOT EXISTS companies (
    ticker TEXT PRIMARY KEY,
    cik TEXT,
    name TEXT,
    sic TEXT,
    sic_description TEXT,
    fiscal_year_end TEXT,
    updated_at TIMESTAMP
)
```

### Filings Table

Stores information about filings:

```sql
CREATE TABLE IF NOT EXISTS filings (
    filing_id TEXT PRIMARY KEY,
    ticker TEXT,
    accession_number TEXT,
    filing_type TEXT,
    filing_date DATE,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    fiscal_period_end_date DATE,
    has_xbrl BOOLEAN,
    updated_at TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES companies(ticker)
)
```

### Financial Facts Table

Stores financial facts extracted from XBRL:

```sql
CREATE TABLE IF NOT EXISTS financial_facts (
    fact_id TEXT PRIMARY KEY,
    filing_id TEXT,
    ticker TEXT,
    xbrl_tag TEXT,
    metric_name TEXT,
    value DOUBLE,
    unit TEXT,
    period_type TEXT,
    start_date DATE,
    end_date DATE,
    segment TEXT,
    context_id TEXT,
    updated_at TIMESTAMP,
    FOREIGN KEY (filing_id) REFERENCES filings(filing_id),
    FOREIGN KEY (ticker) REFERENCES companies(ticker)
)
```

### Time Series Metrics Table

Stores time series metrics for efficient querying:

```sql
CREATE TABLE IF NOT EXISTS time_series_metrics (
    metric_id TEXT PRIMARY KEY,
    ticker TEXT,
    metric_name TEXT,
    period_type TEXT,
    start_date DATE,
    end_date DATE,
    value DOUBLE,
    unit TEXT,
    filing_id TEXT,
    fact_id TEXT,
    updated_at TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES companies(ticker),
    FOREIGN KEY (filing_id) REFERENCES filings(filing_id),
    FOREIGN KEY (fact_id) REFERENCES financial_facts(fact_id)
)
```

## Pipeline Flow

The quantitative ETL pipeline follows these steps:

1. **Download Filing**: Download the SEC filing using the SEC downloader
2. **Extract XBRL Data**: Extract structured financial data from XBRL
3. **Store Company Data**: Store company information in the DuckDB database
4. **Store Filing Data**: Store filing information in the DuckDB database
5. **Store Financial Facts**: Store financial facts in the DuckDB database
6. **Generate Time Series Metrics**: Generate time series metrics for efficient querying

## Usage

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

# Check the result
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Successfully processed filing: {result}")
```

## Configuration

The quantitative pipeline can be configured with the following parameters:

- `db_path`: Path to the DuckDB database file (default: "data/financial_data.duckdb")
- `use_parallel`: Whether to use parallel processing (default: True)
- `num_workers`: Number of workers for parallel processing (default: 4)

Example:

```python
from sec_filing_analyzer.pipeline import QuantitativeETLPipeline

# Initialize quantitative pipeline with custom configuration
pipeline = QuantitativeETLPipeline(
    db_path="data/custom_financial_data.duckdb",
    use_parallel=True,
    num_workers=8
)
```

## Performance Considerations

- **Parallel Processing**: Use parallel processing for processing multiple filings
- **Indexing**: The database automatically creates indexes for efficient querying
- **Batch Processing**: Financial facts are stored in batches for improved performance
- **Memory Usage**: DuckDB is designed to be memory-efficient for analytical queries
