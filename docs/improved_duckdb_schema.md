# Improved DuckDB Schema for Financial Data

This document describes the improved DuckDB schema for storing financial data extracted from SEC filings.

## Overview

The improved schema follows a star schema design, which is optimized for analytical queries. The key improvements include:

1. **Dedicated Metrics Table**: A central repository for all metric definitions
2. **Numeric Primary Keys**: For better performance and storage efficiency
3. **Enhanced Metadata**: More descriptive fields for better documentation
4. **Standardized Naming**: Consistent naming conventions across tables
5. **Prebuilt Views**: For common query patterns

## Schema Structure

### Core Tables

#### companies

Stores information about companies.

| Column | Type | Description |
|--------|------|-------------|
| company_id | INTEGER | Primary key |
| ticker | VARCHAR | Stock ticker symbol (unique) |
| name | VARCHAR | Company name |
| cik | VARCHAR | SEC Central Index Key |
| sic | VARCHAR | Standard Industrial Classification code |
| sector | VARCHAR | Industry sector |
| industry | VARCHAR | Specific industry |
| exchange | VARCHAR | Stock exchange |
| created_at | TIMESTAMP | Record creation timestamp |
| updated_at | TIMESTAMP | Record update timestamp |

#### filings

Stores information about SEC filings.

| Column | Type | Description |
|--------|------|-------------|
| filing_id | INTEGER | Primary key |
| company_id | INTEGER | Foreign key to companies.company_id |
| accession_number | VARCHAR | SEC accession number (unique) |
| filing_type | VARCHAR | Filing type (e.g., 10-K, 10-Q) |
| filing_date | DATE | Date the filing was submitted |
| fiscal_year | INTEGER | Fiscal year |
| fiscal_period | VARCHAR | Fiscal period (e.g., Q1, Q2, FY) |
| fiscal_period_end_date | DATE | End date of the fiscal period |
| document_url | VARCHAR | URL to the filing document |
| has_xbrl | BOOLEAN | Whether the filing has XBRL data |
| created_at | TIMESTAMP | Record creation timestamp |
| updated_at | TIMESTAMP | Record update timestamp |

#### metrics

Stores definitions of financial metrics.

| Column | Type | Description |
|--------|------|-------------|
| metric_id | INTEGER | Primary key |
| metric_name | VARCHAR | Unique identifier for the metric |
| display_name | VARCHAR | Human-readable name |
| description | VARCHAR | Detailed description |
| category | VARCHAR | Category (e.g., income_statement, balance_sheet) |
| unit_of_measure | VARCHAR | Unit of measure (e.g., USD, shares, percent) |
| is_calculated | BOOLEAN | Whether this is a calculated metric |
| calculation_formula | VARCHAR | Formula for calculated metrics |
| created_at | TIMESTAMP | Record creation timestamp |
| updated_at | TIMESTAMP | Record update timestamp |

#### facts

Stores financial facts extracted from filings (core fact table).

| Column | Type | Description |
|--------|------|-------------|
| fact_id | INTEGER | Primary key |
| filing_id | INTEGER | Foreign key to filings.filing_id |
| metric_id | INTEGER | Foreign key to metrics.metric_id |
| value | DOUBLE | Numeric value of the fact |
| as_reported | BOOLEAN | Whether this is the as-reported value |
| normalized_value | DOUBLE | Normalized value (optional) |
| period_type | VARCHAR | Period type (instant, duration) |
| start_date | DATE | Start date for duration facts |
| end_date | DATE | End date for all facts |
| context_id | VARCHAR | Original XBRL context ID |
| decimals | INTEGER | Precision information |
| created_at | TIMESTAMP | Record creation timestamp |
| updated_at | TIMESTAMP | Record update timestamp |

### Supporting Tables

#### xbrl_tag_mappings

Maps XBRL tags to standardized metrics.

| Column | Type | Description |
|--------|------|-------------|
| mapping_id | INTEGER | Primary key |
| xbrl_tag | VARCHAR | XBRL tag (unique) |
| metric_id | INTEGER | Foreign key to metrics.metric_id |
| is_custom | BOOLEAN | Whether this is a custom tag |
| taxonomy | VARCHAR | Taxonomy (e.g., us-gaap, ifrs) |
| taxonomy_version | VARCHAR | Version of the taxonomy |
| created_at | TIMESTAMP | Record creation timestamp |
| updated_at | TIMESTAMP | Record update timestamp |

### Views

#### time_series_view

A view that joins facts, filings, companies, and metrics for time series analysis.

```sql
SELECT 
    c.ticker,
    c.name AS company_name,
    f.fiscal_year,
    f.fiscal_period,
    f.filing_date,
    m.metric_name,
    m.display_name,
    m.category,
    m.unit_of_measure,
    fa.value,
    fa.normalized_value,
    fa.end_date
FROM 
    facts fa
JOIN 
    filings f ON fa.filing_id = f.filing_id
JOIN 
    companies c ON f.company_id = c.company_id
JOIN 
    metrics m ON fa.metric_id = m.metric_id
```

#### company_metrics_view

A view optimized for company-specific queries.

```sql
SELECT 
    c.ticker,
    c.name AS company_name,
    f.fiscal_year,
    f.fiscal_period,
    m.category,
    m.metric_name,
    m.display_name,
    fa.value,
    m.unit_of_measure,
    fa.end_date
FROM 
    facts fa
JOIN 
    filings f ON fa.filing_id = f.filing_id
JOIN 
    companies c ON f.company_id = c.company_id
JOIN 
    metrics m ON fa.metric_id = m.metric_id
ORDER BY 
    c.ticker, f.fiscal_year, f.fiscal_period, m.category, m.metric_name
```

#### company_comparison_view

A view optimized for comparing companies.

```sql
SELECT 
    f.fiscal_year,
    f.fiscal_period,
    m.metric_name,
    m.display_name,
    m.category,
    c.ticker,
    fa.value,
    m.unit_of_measure
FROM 
    facts fa
JOIN 
    filings f ON fa.filing_id = f.filing_id
JOIN 
    companies c ON f.company_id = c.company_id
JOIN 
    metrics m ON fa.metric_id = m.metric_id
ORDER BY 
    f.fiscal_year, f.fiscal_period, m.category, m.metric_name, c.ticker
```

## Migration

To migrate from the old schema to the new schema, use the migration script:

```bash
python src/scripts/migrate_duckdb_schema.py --old-db data/financial_data.duckdb --new-db data/financial_data_new.duckdb
```

## Example Queries

### Get Revenue for a Company Over Time

```sql
SELECT 
    ticker,
    fiscal_year,
    fiscal_period,
    value
FROM 
    company_metrics_view
WHERE 
    ticker = 'MSFT' AND
    metric_name = 'revenue'
ORDER BY 
    fiscal_year, fiscal_period
```

### Compare Revenue Across Companies

```sql
SELECT 
    fiscal_year,
    fiscal_period,
    ticker,
    value
FROM 
    company_comparison_view
WHERE 
    metric_name = 'revenue' AND
    fiscal_year >= 2020
ORDER BY 
    fiscal_year, fiscal_period, ticker
```

### Get Latest Financial Metrics for a Company

```sql
WITH latest_filing AS (
    SELECT 
        MAX(filing_id) AS filing_id
    FROM 
        filings
    WHERE 
        company_id = (SELECT company_id FROM companies WHERE ticker = 'MSFT')
        AND fiscal_period = 'FY'
)
SELECT 
    m.category,
    m.display_name,
    fa.value,
    m.unit_of_measure
FROM 
    facts fa
JOIN 
    metrics m ON fa.metric_id = m.metric_id
WHERE 
    fa.filing_id = (SELECT filing_id FROM latest_filing)
ORDER BY 
    m.category, m.display_name
```

### Calculate Year-over-Year Growth

```sql
WITH revenue_by_year AS (
    SELECT 
        fiscal_year,
        value
    FROM 
        company_metrics_view
    WHERE 
        ticker = 'MSFT' AND
        metric_name = 'revenue' AND
        fiscal_period = 'FY'
    ORDER BY 
        fiscal_year
)
SELECT 
    current.fiscal_year,
    current.value AS current_revenue,
    previous.value AS previous_revenue,
    (current.value - previous.value) / previous.value * 100 AS yoy_growth_percent
FROM 
    revenue_by_year current
JOIN 
    revenue_by_year previous ON current.fiscal_year = previous.fiscal_year + 1
ORDER BY 
    current.fiscal_year
```

### Find Companies with Highest Revenue Growth

```sql
WITH revenue_by_year AS (
    SELECT 
        ticker,
        fiscal_year,
        value
    FROM 
        company_metrics_view
    WHERE 
        metric_name = 'revenue' AND
        fiscal_period = 'FY' AND
        fiscal_year IN (2021, 2022)
),
growth_rates AS (
    SELECT 
        r2022.ticker,
        r2022.value AS revenue_2022,
        r2021.value AS revenue_2021,
        (r2022.value - r2021.value) / r2021.value * 100 AS growth_rate
    FROM 
        revenue_by_year r2022
    JOIN 
        revenue_by_year r2021 ON r2022.ticker = r2021.ticker AND r2022.fiscal_year = 2022 AND r2021.fiscal_year = 2021
)
SELECT 
    g.ticker,
    c.name AS company_name,
    g.revenue_2021,
    g.revenue_2022,
    g.growth_rate
FROM 
    growth_rates g
JOIN 
    companies c ON g.ticker = c.ticker
ORDER BY 
    g.growth_rate DESC
LIMIT 10
```

## Benefits of the New Schema

1. **Better Performance**: Optimized for analytical queries with proper indexing and numeric keys
2. **Improved Data Quality**: Better constraints and relationships
3. **Enhanced Metadata**: More descriptive fields for better documentation
4. **Easier Querying**: Prebuilt views for common query patterns
5. **More Flexibility**: Easier to add new metrics and fact types
6. **Better Documentation**: Self-documenting schema with clear naming conventions
