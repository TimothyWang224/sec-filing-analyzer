-- DuckDB Schema for Financial Data from XBRL

-- Companies table
CREATE TABLE IF NOT EXISTS companies (
    ticker VARCHAR PRIMARY KEY,
    name VARCHAR,
    cik VARCHAR,
    sic VARCHAR,
    sector VARCHAR,
    industry VARCHAR,
    exchange VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Filings table
CREATE TABLE IF NOT EXISTS filings (
    id VARCHAR PRIMARY KEY,
    ticker VARCHAR REFERENCES companies(ticker),
    accession_number VARCHAR,
    filing_type VARCHAR,
    filing_date DATE,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    fiscal_period_end_date DATE,
    document_url VARCHAR,
    has_xbrl BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Financial facts table (normalized)
CREATE TABLE IF NOT EXISTS financial_facts (
    id VARCHAR PRIMARY KEY,
    filing_id VARCHAR REFERENCES filings(id),
    xbrl_tag VARCHAR,
    metric_name VARCHAR,
    value DOUBLE,
    unit VARCHAR,
    period_type VARCHAR,  -- 'instant', 'duration'
    start_date DATE,
    end_date DATE,
    segment VARCHAR,      -- For dimensional data
    context_id VARCHAR,   -- Original XBRL context ID
    decimals INTEGER,     -- Precision information
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (filing_id, xbrl_tag, period_type, context_id)
);

-- Time series metrics (denormalized for analysis)
CREATE TABLE IF NOT EXISTS time_series_metrics (
    ticker VARCHAR REFERENCES companies(ticker),
    metric_name VARCHAR,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    value DOUBLE,
    unit VARCHAR,
    filing_id VARCHAR REFERENCES filings(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, metric_name, fiscal_year, fiscal_quarter)
);

-- Financial ratios (calculated)
CREATE TABLE IF NOT EXISTS financial_ratios (
    ticker VARCHAR REFERENCES companies(ticker),
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    ratio_name VARCHAR,
    value DOUBLE,
    filing_id VARCHAR REFERENCES filings(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, ratio_name, fiscal_year, fiscal_quarter)
);

-- XBRL tag mappings (for standardization)
CREATE TABLE IF NOT EXISTS xbrl_tag_mappings (
    xbrl_tag VARCHAR PRIMARY KEY,
    standard_metric_name VARCHAR,
    category VARCHAR,  -- 'income_statement', 'balance_sheet', 'cash_flow', etc.
    description VARCHAR,
    is_custom BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert standard XBRL tag mappings
INSERT OR IGNORE INTO xbrl_tag_mappings (xbrl_tag, standard_metric_name, category, description) VALUES
-- Income Statement
('Revenues', 'revenue', 'income_statement', 'Total revenue'),
('RevenueFromContractWithCustomerExcludingAssessedTax', 'revenue', 'income_statement', 'Revenue from contracts with customers'),
('SalesRevenueNet', 'revenue', 'income_statement', 'Net sales revenue'),
('SalesRevenueGoodsNet', 'revenue_goods', 'income_statement', 'Net sales revenue from goods'),
('SalesRevenueServicesNet', 'revenue_services', 'income_statement', 'Net sales revenue from services'),
('CostOfRevenue', 'cost_of_revenue', 'income_statement', 'Cost of revenue'),
('CostOfGoodsAndServicesSold', 'cost_of_revenue', 'income_statement', 'Cost of goods and services sold'),
('GrossProfit', 'gross_profit', 'income_statement', 'Gross profit'),
('OperatingExpenses', 'operating_expenses', 'income_statement', 'Total operating expenses'),
('ResearchAndDevelopmentExpense', 'rd_expense', 'income_statement', 'Research and development expense'),
('SellingGeneralAndAdministrativeExpense', 'sga_expense', 'income_statement', 'Selling, general and administrative expense'),
('OperatingIncomeLoss', 'operating_income', 'income_statement', 'Operating income or loss'),
('NonoperatingIncomeExpense', 'nonoperating_income', 'income_statement', 'Nonoperating income or expense'),
('InterestExpense', 'interest_expense', 'income_statement', 'Interest expense'),
('IncomeTaxExpenseBenefit', 'income_tax_expense', 'income_statement', 'Income tax expense or benefit'),
('NetIncomeLoss', 'net_income', 'income_statement', 'Net income or loss'),
('EarningsPerShareBasic', 'eps_basic', 'income_statement', 'Basic earnings per share'),
('EarningsPerShareDiluted', 'eps_diluted', 'income_statement', 'Diluted earnings per share'),
('WeightedAverageNumberOfSharesOutstandingBasic', 'shares_basic', 'income_statement', 'Weighted average shares outstanding - basic'),
('WeightedAverageNumberOfDilutedSharesOutstanding', 'shares_diluted', 'income_statement', 'Weighted average shares outstanding - diluted'),

-- Balance Sheet
('Assets', 'total_assets', 'balance_sheet', 'Total assets'),
('AssetsCurrent', 'current_assets', 'balance_sheet', 'Current assets'),
('CashAndCashEquivalentsAtCarryingValue', 'cash_and_equivalents', 'balance_sheet', 'Cash and cash equivalents'),
('ShortTermInvestments', 'short_term_investments', 'balance_sheet', 'Short-term investments'),
('AccountsReceivableNetCurrent', 'accounts_receivable', 'balance_sheet', 'Accounts receivable, net'),
('InventoryNet', 'inventory', 'balance_sheet', 'Inventory, net'),
('AssetsNoncurrent', 'noncurrent_assets', 'balance_sheet', 'Noncurrent assets'),
('PropertyPlantAndEquipmentNet', 'ppe_net', 'balance_sheet', 'Property, plant and equipment, net'),
('Goodwill', 'goodwill', 'balance_sheet', 'Goodwill'),
('IntangibleAssetsNetExcludingGoodwill', 'intangible_assets', 'balance_sheet', 'Intangible assets, net'),
('LongTermInvestments', 'long_term_investments', 'balance_sheet', 'Long-term investments'),
('Liabilities', 'total_liabilities', 'balance_sheet', 'Total liabilities'),
('LiabilitiesCurrent', 'current_liabilities', 'balance_sheet', 'Current liabilities'),
('AccountsPayableCurrent', 'accounts_payable', 'balance_sheet', 'Accounts payable'),
('AccruedLiabilitiesCurrent', 'accrued_liabilities', 'balance_sheet', 'Accrued liabilities'),
('LongTermDebtCurrent', 'current_long_term_debt', 'balance_sheet', 'Current portion of long-term debt'),
('LiabilitiesNoncurrent', 'noncurrent_liabilities', 'balance_sheet', 'Noncurrent liabilities'),
('LongTermDebtNoncurrent', 'long_term_debt', 'balance_sheet', 'Long-term debt'),
('StockholdersEquity', 'stockholders_equity', 'balance_sheet', 'Total stockholders equity'),
('CommonStockParOrStatedValuePerShare', 'common_stock_par_value', 'balance_sheet', 'Common stock par value'),
('CommonStock', 'common_stock', 'balance_sheet', 'Common stock'),
('AdditionalPaidInCapital', 'additional_paid_in_capital', 'balance_sheet', 'Additional paid-in capital'),
('RetainedEarningsAccumulatedDeficit', 'retained_earnings', 'balance_sheet', 'Retained earnings or accumulated deficit'),
('AccumulatedOtherComprehensiveIncomeLossNetOfTax', 'accumulated_other_comprehensive_income', 'balance_sheet', 'Accumulated other comprehensive income'),
('TreasuryStockValue', 'treasury_stock', 'balance_sheet', 'Treasury stock'),

-- Cash Flow Statement
('NetCashProvidedByUsedInOperatingActivities', 'operating_cash_flow', 'cash_flow', 'Net cash provided by operating activities'),
('NetCashProvidedByUsedInInvestingActivities', 'investing_cash_flow', 'cash_flow', 'Net cash provided by investing activities'),
('NetCashProvidedByUsedInFinancingActivities', 'financing_cash_flow', 'cash_flow', 'Net cash provided by financing activities'),
('DepreciationDepletionAndAmortization', 'depreciation_amortization', 'cash_flow', 'Depreciation, depletion and amortization'),
('PaymentsToAcquirePropertyPlantAndEquipment', 'capex', 'cash_flow', 'Capital expenditures'),
('ProceedsFromIssuanceOfCommonStock', 'stock_issuance_proceeds', 'cash_flow', 'Proceeds from issuance of common stock'),
('PaymentsOfDividends', 'dividends_paid', 'cash_flow', 'Dividends paid'),
('PaymentsForRepurchaseOfCommonStock', 'stock_repurchase', 'cash_flow', 'Payments for repurchase of common stock'),
('CashAndCashEquivalentsPeriodIncreaseDecrease', 'net_change_in_cash', 'cash_flow', 'Net change in cash and cash equivalents');
