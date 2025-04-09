# XBRL to DuckDB Integration

This document describes how we leverage the edgar library's XBRL parsing capabilities to extract financial data from SEC filings and store it in a DuckDB database.

## Overview

The SEC Filing Analyzer uses the edgar library to extract XBRL data from SEC filings and store it in a structured DuckDB database. This allows for efficient querying and analysis of financial data across multiple companies and time periods.

## Components

### 1. EdgarXBRLToDuckDBExtractor

The `EdgarXBRLToDuckDBExtractor` class is the main component responsible for extracting XBRL data from SEC filings and storing it in the DuckDB database. It leverages the edgar library's built-in XBRL parsing capabilities to extract financial data, including:

- Financial statements (balance sheet, income statement, cash flow statement)
- US-GAAP facts
- Fiscal period information
- Company metadata

### 2. OptimizedDuckDBStore

The `OptimizedDuckDBStore` class provides an optimized interface to store and query financial data in a DuckDB database. It supports both individual and batch operations for storing:

- Companies
- Filings
- Financial facts
- Time series metrics
- Financial ratios

### 3. Database Schema

The DuckDB database uses the following schema:

- **companies**: Stores company information (ticker, name, CIK, etc.)
- **filings**: Stores filing metadata (accession number, filing type, fiscal period, etc.)
- **financial_facts**: Stores individual financial facts extracted from XBRL data
- **time_series_metrics**: Stores time series metrics for easy querying across time periods
- **financial_ratios**: Stores calculated financial ratios
- **financial_data**: Stores additional financial data

## Usage

### Extracting Data from a Single Filing

```python
from sec_filing_analyzer.data_processing.edgar_xbrl_to_duckdb import EdgarXBRLToDuckDBExtractor

# Create the extractor
extractor = EdgarXBRLToDuckDBExtractor(db_path="data/financial_data.duckdb")

# Process a filing
result = extractor.process_filing(
    ticker="MSFT",
    accession_number="0001564590-22-026876"  # Microsoft's 10-K from July 2022
)

print(f"Successfully processed filing: {result['message']}")
```

### Processing Multiple Filings for a Company

```python
# Process multiple filings for a company
result = extractor.process_company_filings(
    ticker="MSFT",
    filing_types=["10-K", "10-Q"],
    limit=5  # Process up to 5 filings
)

print(f"Successfully processed filings: {result['message']}")
```

### Processing Multiple Companies

```python
# Process filings for multiple companies
result = extractor.process_multiple_companies(
    tickers=["MSFT", "AAPL", "GOOGL"],
    filing_types=["10-K", "10-Q"],
    limit_per_company=3  # Process up to 3 filings per company
)

print(f"Successfully processed companies: {result['message']}")
```

## Querying the Database

The DuckDB database can be queried using SQL. Here are some example queries:

### Get Financial Metrics for a Company

```python
# Get financial metrics for a company
metrics = extractor.db.get_company_metrics(
    ticker="MSFT",
    metrics=["revenue", "net_income", "total_assets"],
    fiscal_years=[2020, 2021, 2022]
)

print(metrics)
```

### Compare Companies

```python
# Compare companies on a specific metric
comparison = extractor.db.compare_companies(
    tickers=["MSFT", "AAPL", "GOOGL"],
    metric="revenue",
    fiscal_years=[2020, 2021, 2022]
)

print(comparison)
```

## Benefits of Using Edgar's XBRL Parsing

By leveraging the edgar library's built-in XBRL parsing capabilities, we gain several benefits:

1. **Robust XBRL Parsing**: The edgar library handles the complexities of XBRL documents, including namespaces, dimensions, and linkbases.

2. **Access to Financial Statements**: The library provides high-level access to financial statements with proper labels and hierarchical organization.

3. **Direct Access to US-GAAP Facts**: We can directly query US-GAAP facts without needing to manually parse XBRL.

4. **Fiscal Period Information**: The library extracts fiscal period information from DEI facts.

5. **Comprehensive Data Extraction**: We can extract a wide range of financial data, including statements, facts, and metadata.

## Conclusion

The integration of edgar's XBRL parsing capabilities with our DuckDB database provides a powerful solution for extracting, storing, and analyzing financial data from SEC filings. This approach allows us to efficiently process large volumes of financial data and perform complex analyses across multiple companies and time periods.
