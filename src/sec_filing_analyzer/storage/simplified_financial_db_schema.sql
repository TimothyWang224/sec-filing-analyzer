-- Simplified DuckDB Schema for Financial Data from XBRL

-- Companies table
CREATE TABLE IF NOT EXISTS companies (
    company_id INTEGER PRIMARY KEY,
    ticker VARCHAR UNIQUE NOT NULL,
    name VARCHAR,
    cik VARCHAR,
    sic VARCHAR,
    sector VARCHAR,
    industry VARCHAR,
    exchange VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Filings table
CREATE TABLE IF NOT EXISTS filings (
    filing_id INTEGER PRIMARY KEY,
    company_id INTEGER REFERENCES companies(company_id),
    accession_number VARCHAR UNIQUE NOT NULL,
    filing_type VARCHAR,
    filing_date DATE,
    fiscal_year INTEGER,
    fiscal_period VARCHAR,
    fiscal_period_end_date DATE,
    document_url VARCHAR,
    has_xbrl BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Metrics table
CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY,
    metric_name VARCHAR UNIQUE NOT NULL,
    display_name VARCHAR,
    description VARCHAR,
    category VARCHAR,
    unit_of_measure VARCHAR,
    is_calculated BOOLEAN DEFAULT FALSE,
    calculation_formula VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Facts table
CREATE TABLE IF NOT EXISTS facts (
    fact_id INTEGER PRIMARY KEY,
    filing_id INTEGER REFERENCES filings(filing_id),
    metric_id INTEGER REFERENCES metrics(metric_id),
    value DOUBLE,
    as_reported BOOLEAN DEFAULT TRUE,
    normalized_value DOUBLE,
    period_type VARCHAR,
    start_date DATE,
    end_date DATE,
    context_id VARCHAR,
    decimals INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (filing_id, metric_id, context_id)
);

-- XBRL tag mappings
CREATE TABLE IF NOT EXISTS xbrl_tag_mappings (
    mapping_id INTEGER PRIMARY KEY,
    xbrl_tag VARCHAR UNIQUE NOT NULL,
    metric_id INTEGER REFERENCES metrics(metric_id),
    is_custom BOOLEAN DEFAULT FALSE,
    taxonomy VARCHAR,
    taxonomy_version VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Time series view
CREATE VIEW IF NOT EXISTS time_series_view AS
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
    metrics m ON fa.metric_id = m.metric_id;

-- Company metrics view
CREATE VIEW IF NOT EXISTS company_metrics_view AS
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
    c.ticker, f.fiscal_year, f.fiscal_period, m.category, m.metric_name;

-- Company comparison view
CREATE VIEW IF NOT EXISTS company_comparison_view AS
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
    f.fiscal_year, f.fiscal_period, m.category, m.metric_name, c.ticker;
