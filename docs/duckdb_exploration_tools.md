# DuckDB Exploration Tools

This document provides information on various tools for exploring DuckDB databases used in the SEC Filing Analyzer.

## Overview

DuckDB is an in-process SQL OLAP database management system that provides efficient analytical queries on structured data. The SEC Filing Analyzer uses DuckDB to store structured financial data extracted from XBRL filings.

## Browser-Based Tools

### 1. DuckDB Web Explorer

The DuckDB Web Explorer is a simple browser-based interface for exploring DuckDB databases. It provides a SQL editor, table browser, and schema viewer.

#### Installation

No additional installation is required. The tool is included in the SEC Filing Analyzer.

#### Usage

```bash
# Launch the DuckDB Web Explorer
python src/scripts/launch_duckdb_web.py --db data/financial_data.duckdb
```

This will open a web browser with the DuckDB Web Explorer interface. You can:

- Run SQL queries
- Browse tables
- View table schemas
- Export query results

### 2. Streamlit DuckDB Explorer

The Streamlit DuckDB Explorer is a more feature-rich browser-based interface for exploring DuckDB databases. It provides data visualization, pagination, and more.

#### Installation

```bash
# Install required packages
pip install -r requirements-tools.txt
```

#### Usage

```bash
# Launch the Streamlit DuckDB Explorer
streamlit run src/scripts/streamlit_duckdb_explorer.py
```

This will open a web browser with the Streamlit DuckDB Explorer interface. You can:

- Browse tables
- View table schemas
- Run SQL queries
- Visualize data with charts
- Export query results

### 3. Jupyter Notebook

The SEC Filing Analyzer includes a Jupyter notebook for exploring DuckDB databases.

#### Installation

```bash
# Install required packages
pip install -r requirements-tools.txt
```

#### Usage

```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/explore_duckdb.ipynb
```

This will open a web browser with the Jupyter notebook interface. The notebook includes examples of:

- Listing tables
- Exploring table schemas
- Viewing sample data
- Running custom queries
- Analyzing financial data
- Visualizing financial data
- Comparing companies

## Desktop Tools

### 1. DBeaver

DBeaver is a free, open-source universal database tool that supports DuckDB. It provides a rich GUI for database exploration.

#### Installation

1. Download DBeaver from [https://dbeaver.io/](https://dbeaver.io/)
2. Install DBeaver following the instructions for your operating system

#### Usage

1. Launch DBeaver
2. Create a new connection:
   - Click "New Database Connection"
   - Search for "DuckDB" in the database selection dialog
   - Configure the connection with your database file path
   - Test the connection and finish
3. Explore the database:
   - Browse tables
   - View and edit data
   - Run SQL queries
   - Create ER diagrams
   - Export/import data

### 2. TablePlus

TablePlus is a modern, native tool for database management that supports DuckDB.

#### Installation

1. Download TablePlus from [https://tableplus.com/](https://tableplus.com/)
2. Install TablePlus following the instructions for your operating system

#### Usage

1. Launch TablePlus
2. Create a new connection:
   - Click "Create a new connection"
   - Select "DuckDB"
   - Configure the connection with your database file path
   - Test the connection and connect
3. Explore the database:
   - Browse tables
   - View and edit data
   - Run SQL queries
   - Export/import data

## Command-Line Tools

### 1. DuckDB CLI

DuckDB includes a command-line interface for interacting with DuckDB databases.

#### Installation

No additional installation is required if you have DuckDB installed.

#### Usage

```bash
# Launch the DuckDB CLI
duckdb data/financial_data.duckdb
```

This will open the DuckDB CLI. You can:

- Run SQL queries
- Export query results
- Import data

### 2. DuckDB Query Script

The SEC Filing Analyzer includes a simple script for running SQL queries against DuckDB databases.

#### Usage

```bash
# Run a SQL query
python src/scripts/query_duckdb.py --query "SELECT * FROM companies LIMIT 5"

# Run a SQL query from a file
python src/scripts/query_duckdb.py --query-file queries/companies.sql

# Output results in CSV format
python src/scripts/query_duckdb.py --query "SELECT * FROM companies" --format csv --output companies.csv

# Interactive mode
python src/scripts/query_duckdb.py
```

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
