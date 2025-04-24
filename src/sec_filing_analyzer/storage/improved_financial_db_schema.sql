-- Improved DuckDB Schema for Financial Data from XBRL

-- Companies table
CREATE TABLE IF NOT EXISTS companies (
    company_id INTEGER PRIMARY KEY,  -- Numeric ID for better performance
    ticker VARCHAR UNIQUE NOT NULL,  -- Still need ticker as a unique identifier
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
    filing_id INTEGER PRIMARY KEY,  -- Numeric ID for better performance
    company_id INTEGER REFERENCES companies(company_id),
    accession_number VARCHAR UNIQUE NOT NULL,  -- SEC accession number as unique identifier
    filing_type VARCHAR,  -- e.g., 10-K, 10-Q
    filing_date DATE,  -- The actual date filed
    fiscal_year INTEGER,
    fiscal_period VARCHAR,  -- e.g., 'Q1', 'Q2', 'FY'
    fiscal_period_end_date DATE,
    document_url VARCHAR,
    has_xbrl BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Metrics table (new)
CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY,
    metric_name VARCHAR UNIQUE NOT NULL,
    display_name VARCHAR,  -- Human-readable name
    description VARCHAR,  -- Detailed description
    category VARCHAR,  -- e.g., 'income_statement', 'balance_sheet', 'cash_flow', 'ratio'
    unit_of_measure VARCHAR,  -- e.g., 'USD', 'shares', 'percent'
    is_calculated BOOLEAN DEFAULT FALSE,  -- Whether this is a calculated metric
    calculation_formula VARCHAR,  -- Formula for calculated metrics
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Facts table (core fact table in star schema)
CREATE TABLE IF NOT EXISTS facts (
    fact_id INTEGER PRIMARY KEY,
    filing_id INTEGER REFERENCES filings(filing_id),
    metric_id INTEGER REFERENCES metrics(metric_id),
    value DOUBLE,
    as_reported BOOLEAN DEFAULT TRUE,  -- Whether this is the as-reported value or normalized
    normalized_value DOUBLE,  -- Optional normalized value
    period_type VARCHAR,  -- 'instant', 'duration'
    start_date DATE,
    end_date DATE,
    context_id VARCHAR,  -- Original XBRL context ID
    decimals INTEGER,  -- Precision information
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (filing_id, metric_id, context_id)  -- Ensure no duplicate facts
);

-- XBRL tag mappings (for standardization)
CREATE TABLE IF NOT EXISTS xbrl_tag_mappings (
    mapping_id INTEGER PRIMARY KEY,
    xbrl_tag VARCHAR UNIQUE NOT NULL,
    metric_id INTEGER REFERENCES metrics(metric_id),
    is_custom BOOLEAN DEFAULT FALSE,
    taxonomy VARCHAR,  -- e.g., 'us-gaap', 'ifrs', 'custom'
    taxonomy_version VARCHAR,  -- Version of the taxonomy
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Time series view (for easier querying)
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

-- Company metrics view (for easier company-specific querying)
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

-- Comparison view (for comparing companies)
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

-- Insert standard metrics
INSERT OR IGNORE INTO metrics (metric_id, metric_name, display_name, description, category, unit_of_measure) VALUES
-- Income Statement
(1, 'revenue', 'Revenue', 'Total revenue from operations', 'income_statement', 'USD'),
(2, 'cost_of_revenue', 'Cost of Revenue', 'Direct costs attributable to the production of goods or services', 'income_statement', 'USD'),
(3, 'gross_profit', 'Gross Profit', 'Revenue minus cost of revenue', 'income_statement', 'USD'),
(4, 'operating_expenses', 'Operating Expenses', 'Total operating expenses', 'income_statement', 'USD'),
(5, 'operating_income', 'Operating Income', 'Profit from operations before interest and taxes', 'income_statement', 'USD'),
(6, 'net_income', 'Net Income', 'Bottom-line earnings after all expenses', 'income_statement', 'USD'),
(7, 'eps_basic', 'EPS (Basic)', 'Basic earnings per share', 'income_statement', 'USD/share'),
(8, 'eps_diluted', 'EPS (Diluted)', 'Diluted earnings per share', 'income_statement', 'USD/share'),

-- Balance Sheet
(101, 'cash_and_equivalents', 'Cash and Equivalents', 'Cash and short-term, highly liquid investments', 'balance_sheet', 'USD'),
(102, 'short_term_investments', 'Short-term Investments', 'Investments expected to be converted to cash within a year', 'balance_sheet', 'USD'),
(103, 'accounts_receivable', 'Accounts Receivable', 'Money owed to the company by customers', 'balance_sheet', 'USD'),
(104, 'inventory', 'Inventory', 'Goods available for sale or raw materials', 'balance_sheet', 'USD'),
(105, 'total_current_assets', 'Total Current Assets', 'Assets expected to be converted to cash within a year', 'balance_sheet', 'USD'),
(106, 'property_plant_equipment', 'Property, Plant & Equipment', 'Long-term tangible assets', 'balance_sheet', 'USD'),
(107, 'goodwill', 'Goodwill', 'Premium paid over the fair value of acquired assets', 'balance_sheet', 'USD'),
(108, 'intangible_assets', 'Intangible Assets', 'Non-physical assets like patents and trademarks', 'balance_sheet', 'USD'),
(109, 'total_assets', 'Total Assets', 'Sum of all assets', 'balance_sheet', 'USD'),
(110, 'accounts_payable', 'Accounts Payable', 'Money owed by the company to suppliers', 'balance_sheet', 'USD'),
(111, 'short_term_debt', 'Short-term Debt', 'Debt due within one year', 'balance_sheet', 'USD'),
(112, 'total_current_liabilities', 'Total Current Liabilities', 'Obligations due within one year', 'balance_sheet', 'USD'),
(113, 'long_term_debt', 'Long-term Debt', 'Debt due beyond one year', 'balance_sheet', 'USD'),
(114, 'total_liabilities', 'Total Liabilities', 'Sum of all liabilities', 'balance_sheet', 'USD'),
(115, 'stockholders_equity', 'Stockholders Equity', 'Total assets minus total liabilities', 'balance_sheet', 'USD'),

-- Cash Flow
(201, 'operating_cash_flow', 'Operating Cash Flow', 'Net cash from operating activities', 'cash_flow', 'USD'),
(202, 'capital_expenditures', 'Capital Expenditures', 'Funds used to acquire or upgrade assets', 'cash_flow', 'USD'),
(203, 'free_cash_flow', 'Free Cash Flow', 'Operating cash flow minus capital expenditures', 'cash_flow', 'USD'),
(204, 'investing_cash_flow', 'Investing Cash Flow', 'Net cash from investing activities', 'cash_flow', 'USD'),
(205, 'financing_cash_flow', 'Financing Cash Flow', 'Net cash from financing activities', 'cash_flow', 'USD'),
(206, 'net_change_in_cash', 'Net Change in Cash', 'Net increase or decrease in cash', 'cash_flow', 'USD'),

-- Financial Ratios (calculated metrics)
(301, 'gross_margin', 'Gross Margin', 'Gross profit divided by revenue', 'ratio', 'percent'),
(302, 'operating_margin', 'Operating Margin', 'Operating income divided by revenue', 'ratio', 'percent'),
(303, 'net_margin', 'Net Margin', 'Net income divided by revenue', 'ratio', 'percent'),
(304, 'return_on_assets', 'Return on Assets', 'Net income divided by total assets', 'ratio', 'percent'),
(305, 'return_on_equity', 'Return on Equity', 'Net income divided by stockholders'' equity', 'ratio', 'percent'),
(306, 'current_ratio', 'Current Ratio', 'Current assets divided by current liabilities', 'ratio', 'ratio'),
(307, 'debt_to_equity', 'Debt to Equity', 'Total debt divided by stockholders'' equity', 'ratio', 'ratio'),
(308, 'interest_coverage', 'Interest Coverage', 'EBIT divided by interest expense', 'ratio', 'ratio');

-- Insert XBRL tag mappings for common tags
INSERT OR IGNORE INTO xbrl_tag_mappings (mapping_id, xbrl_tag, metric_id, taxonomy) VALUES
(1, 'Revenues', 1, 'us-gaap'),
(2, 'RevenueFromContractWithCustomerExcludingAssessedTax', 1, 'us-gaap'),
(3, 'SalesRevenueNet', 1, 'us-gaap'),
(4, 'CostOfRevenue', 2, 'us-gaap'),
(5, 'CostOfGoodsAndServicesSold', 2, 'us-gaap'),
(6, 'GrossProfit', 3, 'us-gaap'),
(7, 'OperatingExpenses', 4, 'us-gaap'),
(8, 'OperatingIncomeLoss', 5, 'us-gaap'),
(9, 'NetIncomeLoss', 6, 'us-gaap'),
(10, 'EarningsPerShareBasic', 7, 'us-gaap'),
(11, 'EarningsPerShareDiluted', 8, 'us-gaap'),
(12, 'CashAndCashEquivalentsAtCarryingValue', 101, 'us-gaap'),
(13, 'ShortTermInvestments', 102, 'us-gaap'),
(14, 'AccountsReceivableNetCurrent', 103, 'us-gaap'),
(15, 'InventoryNet', 104, 'us-gaap'),
(16, 'AssetsCurrent', 105, 'us-gaap'),
(17, 'PropertyPlantAndEquipmentNet', 106, 'us-gaap'),
(18, 'Goodwill', 107, 'us-gaap'),
(19, 'IntangibleAssetsNetExcludingGoodwill', 108, 'us-gaap'),
(20, 'Assets', 109, 'us-gaap'),
(21, 'AccountsPayableCurrent', 110, 'us-gaap'),
(22, 'ShortTermDebt', 111, 'us-gaap'),
(23, 'LiabilitiesCurrent', 112, 'us-gaap'),
(24, 'LongTermDebt', 113, 'us-gaap'),
(25, 'Liabilities', 114, 'us-gaap'),
(26, 'StockholdersEquity', 115, 'us-gaap'),
(27, 'NetCashProvidedByUsedInOperatingActivities', 201, 'us-gaap'),
(28, 'PaymentsToAcquirePropertyPlantAndEquipment', 202, 'us-gaap'),
(29, 'NetCashProvidedByUsedInInvestingActivities', 204, 'us-gaap'),
(30, 'NetCashProvidedByUsedInFinancingActivities', 205, 'us-gaap'),
(31, 'CashAndCashEquivalentsPeriodIncreaseDecrease', 206, 'us-gaap');
