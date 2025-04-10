# Exploring DuckDB Database

This document provides information on how to explore the DuckDB database used for storing quantitative financial data in the SEC Filing Analyzer.

## Overview

The SEC Filing Analyzer uses DuckDB to store structured financial data extracted from XBRL filings. DuckDB is an in-process SQL OLAP database management system that provides efficient analytical queries on structured data.

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

## Exploration Tools

The SEC Filing Analyzer provides several tools for exploring the DuckDB database:

### 1. DuckDB Explorer Script

The `explore_duckdb.py` script provides a simple interface to explore the tables in the DuckDB database:

```bash
# List all tables in the database
python src/scripts/explore_duckdb.py --list-tables

# Describe a specific table
python src/scripts/explore_duckdb.py --describe companies

# Show a sample of rows from a table
python src/scripts/explore_duckdb.py --sample financial_facts --limit 5

# Show table relationships
python src/scripts/explore_duckdb.py --relationships

# Run a custom SQL query
python src/scripts/explore_duckdb.py --query "SELECT * FROM companies LIMIT 5"

# Interactive mode
python src/scripts/explore_duckdb.py
```

### 2. DuckDB Query Script

The `query_duckdb.py` script allows you to run SQL queries against the DuckDB database:

```bash
# Run a SQL query
python src/scripts/query_duckdb.py --query "SELECT * FROM companies LIMIT 5"

# Run a SQL query from a file
python src/scripts/query_duckdb.py --query-file queries/companies.sql

# Output results in CSV format
python src/scripts/query_duckdb.py --query "SELECT * FROM companies" --format csv --output companies.csv

# Output results in JSON format
python src/scripts/query_duckdb.py --query "SELECT * FROM companies" --format json --output companies.json

# Interactive mode
python src/scripts/query_duckdb.py
```

### 3. Jupyter Notebook

The `explore_duckdb.ipynb` notebook provides an interactive way to explore the DuckDB database:

```bash
# Start Jupyter Notebook
jupyter notebook notebooks/explore_duckdb.ipynb
```

The notebook includes examples of:
- Listing tables
- Exploring table schemas
- Viewing sample data
- Running custom queries
- Analyzing financial data
- Visualizing financial data
- Comparing companies

## Example Queries

Here are some example queries you can run against the DuckDB database:

### List all companies

```sql
SELECT * FROM companies
```

### List all filings for a company

```sql
SELECT * FROM filings WHERE ticker = 'MSFT'
```

### Get revenue for a company over time

```sql
SELECT 
    ticker,
    end_date,
    value
FROM 
    time_series_metrics
WHERE 
    ticker = 'MSFT' AND
    metric_name = 'Revenue' AND
    period_type = 'yearly'
ORDER BY 
    end_date
```

### Compare revenue for multiple companies

```sql
SELECT 
    ticker,
    end_date,
    value
FROM 
    time_series_metrics
WHERE 
    metric_name = 'Revenue' AND
    period_type = 'yearly' AND
    ticker IN ('MSFT', 'AAPL', 'GOOGL')
ORDER BY 
    ticker, end_date
```

### Calculate year-over-year growth

```sql
WITH revenue AS (
    SELECT 
        ticker,
        end_date,
        value
    FROM 
        time_series_metrics
    WHERE 
        metric_name = 'Revenue' AND
        period_type = 'yearly' AND
        ticker = 'MSFT'
    ORDER BY 
        end_date
)
SELECT 
    r1.ticker,
    r1.end_date,
    r1.value AS current_revenue,
    r2.value AS previous_revenue,
    (r1.value - r2.value) / r2.value * 100 AS yoy_growth_percent
FROM 
    revenue r1
JOIN 
    revenue r2 ON r1.ticker = r2.ticker AND 
                 EXTRACT(YEAR FROM r1.end_date) = EXTRACT(YEAR FROM r2.end_date) + 1
ORDER BY 
    r1.end_date
```

### Get the latest financial metrics for a company

```sql
SELECT 
    ticker,
    metric_name,
    value,
    end_date
FROM 
    time_series_metrics
WHERE 
    ticker = 'MSFT' AND
    period_type = 'yearly' AND
    metric_name IN ('Revenue', 'NetIncome', 'TotalAssets', 'TotalLiabilities')
ORDER BY 
    end_date DESC
LIMIT 
    4
```

## Direct DuckDB Access

You can also access the DuckDB database directly using the DuckDB Python API:

```python
import duckdb

# Connect to the database
conn = duckdb.connect("data/financial_data.duckdb")

# Run a query
result = conn.execute("SELECT * FROM companies").fetchdf()
print(result)

# Close the connection
conn.close()
```

## Performance Tips

- **Use indexes**: DuckDB automatically creates indexes for primary keys and foreign keys
- **Filter early**: Apply filters early in your queries to reduce the amount of data processed
- **Use appropriate data types**: DuckDB performs better when using appropriate data types
- **Use prepared statements**: For repeated queries, use prepared statements to improve performance
- **Use EXPLAIN**: Use the EXPLAIN statement to understand query execution plans
