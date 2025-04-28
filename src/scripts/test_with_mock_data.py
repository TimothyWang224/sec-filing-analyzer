"""
Test the improved DuckDB store with mock data.
"""

import logging
import os

import duckdb
from rich.console import Console

from sec_filing_analyzer.storage.improved_duckdb_store import ImprovedDuckDBStore

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up console
console = Console()


def create_test_database(db_path):
    """Create a test database with mock data."""
    print(f"Creating test database at {db_path}...")

    # Create the database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Connect to the database
    print(f"Connecting to {db_path}...")
    conn = duckdb.connect(db_path)

    # Create tables
    print("Creating tables...")

    # Companies table
    conn.execute("""
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
        )
    """)

    # Filings table
    conn.execute("""
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
        )
    """)

    # Metrics table
    conn.execute("""
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
        )
    """)

    # Facts table
    conn.execute("""
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
        )
    """)

    # XBRL tag mappings
    conn.execute("""
        CREATE TABLE IF NOT EXISTS xbrl_tag_mappings (
            mapping_id INTEGER PRIMARY KEY,
            xbrl_tag VARCHAR UNIQUE NOT NULL,
            metric_id INTEGER REFERENCES metrics(metric_id),
            is_custom BOOLEAN DEFAULT FALSE,
            taxonomy VARCHAR,
            taxonomy_version VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create views
    print("Creating views...")

    # Time series view
    conn.execute("""
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
            metrics m ON fa.metric_id = m.metric_id
    """)

    # Company metrics view
    conn.execute("""
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
            c.ticker, f.fiscal_year, f.fiscal_period, m.category, m.metric_name
    """)

    # Company comparison view
    conn.execute("""
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
            f.fiscal_year, f.fiscal_period, m.category, m.metric_name, c.ticker
    """)

    # Insert mock data
    print("Inserting mock data...")

    # Insert companies
    conn.execute("""
        INSERT INTO companies (company_id, ticker, name, cik, sic, sector, industry, exchange)
        VALUES 
        (1, 'MSFT', 'Microsoft Corporation', '0000789019', '7372', 'Technology', 'Software', 'NASDAQ'),
        (2, 'AAPL', 'Apple Inc.', '0000320193', '3571', 'Technology', 'Hardware', 'NASDAQ'),
        (3, 'GOOGL', 'Alphabet Inc.', '0001652044', '7370', 'Technology', 'Internet', 'NASDAQ')
    """)

    # Insert filings
    conn.execute("""
        INSERT INTO filings (
            filing_id, company_id, accession_number, filing_type, filing_date, 
            fiscal_year, fiscal_period, fiscal_period_end_date, document_url, has_xbrl
        )
        VALUES 
        (1, 1, '0000789019-22-000010', '10-K', '2022-07-28', 2022, 'FY', '2022-06-30', 'https://www.sec.gov/Archives/edgar/data/789019/000078901922000010/msft-20220630.htm', TRUE),
        (2, 1, '0000789019-22-000004', '10-Q', '2022-01-25', 2022, 'Q2', '2021-12-31', 'https://www.sec.gov/Archives/edgar/data/789019/000078901922000004/msft-20211231.htm', TRUE),
        (3, 2, '0000320193-22-000108', '10-K', '2022-10-28', 2022, 'FY', '2022-09-24', 'https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220924.htm', TRUE)
    """)

    # Insert metrics
    conn.execute("""
        INSERT INTO metrics (
            metric_id, metric_name, display_name, description, category, unit_of_measure, is_calculated
        )
        VALUES 
        (1, 'revenue', 'Revenue', 'Total revenue from operations', 'income_statement', 'USD', FALSE),
        (2, 'net_income', 'Net Income', 'Bottom-line earnings after all expenses', 'income_statement', 'USD', FALSE),
        (3, 'total_assets', 'Total Assets', 'Sum of all assets', 'balance_sheet', 'USD', FALSE),
        (4, 'total_liabilities', 'Total Liabilities', 'Sum of all liabilities', 'balance_sheet', 'USD', FALSE),
        (5, 'stockholders_equity', 'Stockholders Equity', 'Total assets minus total liabilities', 'balance_sheet', 'USD', FALSE),
        (6, 'operating_cash_flow', 'Operating Cash Flow', 'Net cash from operating activities', 'cash_flow', 'USD', FALSE),
        (7, 'eps_basic', 'EPS (Basic)', 'Basic earnings per share', 'income_statement', 'USD/share', FALSE),
        (8, 'eps_diluted', 'EPS (Diluted)', 'Diluted earnings per share', 'income_statement', 'USD/share', FALSE)
    """)

    # Insert facts
    conn.execute("""
        INSERT INTO facts (
            fact_id, filing_id, metric_id, value, as_reported, normalized_value,
            period_type, start_date, end_date, context_id, decimals
        )
        VALUES 
        -- Microsoft 10-K 2022
        (1, 1, 1, 198270000000, TRUE, 198270000000, 'duration', '2021-07-01', '2022-06-30', 'FY2022', -6),
        (2, 1, 2, 72738000000, TRUE, 72738000000, 'duration', '2021-07-01', '2022-06-30', 'FY2022', -6),
        (3, 1, 3, 364840000000, TRUE, 364840000000, 'instant', NULL, '2022-06-30', 'AsOf2022-06-30', -6),
        (4, 1, 4, 198300000000, TRUE, 198300000000, 'instant', NULL, '2022-06-30', 'AsOf2022-06-30', -6),
        (5, 1, 5, 166540000000, TRUE, 166540000000, 'instant', NULL, '2022-06-30', 'AsOf2022-06-30', -6),
        (6, 1, 6, 89299000000, TRUE, 89299000000, 'duration', '2021-07-01', '2022-06-30', 'FY2022', -6),
        (7, 1, 7, 9.70, TRUE, 9.70, 'duration', '2021-07-01', '2022-06-30', 'FY2022', 2),
        (8, 1, 8, 9.65, TRUE, 9.65, 'duration', '2021-07-01', '2022-06-30', 'FY2022', 2),
        
        -- Microsoft 10-Q 2022 Q2
        (9, 2, 1, 51728000000, TRUE, 51728000000, 'duration', '2021-10-01', '2021-12-31', 'Q2FY2022', -6),
        (10, 2, 2, 18765000000, TRUE, 18765000000, 'duration', '2021-10-01', '2021-12-31', 'Q2FY2022', -6),
        (11, 2, 3, 340374000000, TRUE, 340374000000, 'instant', NULL, '2021-12-31', 'AsOf2021-12-31', -6),
        (12, 2, 4, 193549000000, TRUE, 193549000000, 'instant', NULL, '2021-12-31', 'AsOf2021-12-31', -6),
        (13, 2, 5, 146825000000, TRUE, 146825000000, 'instant', NULL, '2021-12-31', 'AsOf2021-12-31', -6),
        (14, 2, 6, 14480000000, TRUE, 14480000000, 'duration', '2021-10-01', '2021-12-31', 'Q2FY2022', -6),
        (15, 2, 7, 2.50, TRUE, 2.50, 'duration', '2021-10-01', '2021-12-31', 'Q2FY2022', 2),
        (16, 2, 8, 2.48, TRUE, 2.48, 'duration', '2021-10-01', '2021-12-31', 'Q2FY2022', 2),
        
        -- Apple 10-K 2022
        (17, 3, 1, 394328000000, TRUE, 394328000000, 'duration', '2021-09-26', '2022-09-24', 'FY2022', -6),
        (18, 3, 2, 99803000000, TRUE, 99803000000, 'duration', '2021-09-26', '2022-09-24', 'FY2022', -6),
        (19, 3, 3, 352755000000, TRUE, 352755000000, 'instant', NULL, '2022-09-24', 'AsOf2022-09-24', -6),
        (20, 3, 4, 302083000000, TRUE, 302083000000, 'instant', NULL, '2022-09-24', 'AsOf2022-09-24', -6),
        (21, 3, 5, 50672000000, TRUE, 50672000000, 'instant', NULL, '2022-09-24', 'AsOf2022-09-24', -6),
        (22, 3, 6, 122151000000, TRUE, 122151000000, 'duration', '2021-09-26', '2022-09-24', 'FY2022', -6),
        (23, 3, 7, 6.15, TRUE, 6.15, 'duration', '2021-09-26', '2022-09-24', 'FY2022', 2),
        (24, 3, 8, 6.11, TRUE, 6.11, 'duration', '2021-09-26', '2022-09-24', 'FY2022', 2)
    """)

    # Insert XBRL tag mappings
    conn.execute("""
        INSERT INTO xbrl_tag_mappings (
            mapping_id, xbrl_tag, metric_id, is_custom, taxonomy
        )
        VALUES 
        (1, 'Revenues', 1, FALSE, 'us-gaap'),
        (2, 'RevenueFromContractWithCustomerExcludingAssessedTax', 1, FALSE, 'us-gaap'),
        (3, 'SalesRevenueNet', 1, FALSE, 'us-gaap'),
        (4, 'NetIncomeLoss', 2, FALSE, 'us-gaap'),
        (5, 'Assets', 3, FALSE, 'us-gaap'),
        (6, 'Liabilities', 4, FALSE, 'us-gaap'),
        (7, 'StockholdersEquity', 5, FALSE, 'us-gaap'),
        (8, 'NetCashProvidedByUsedInOperatingActivities', 6, FALSE, 'us-gaap'),
        (9, 'EarningsPerShareBasic', 7, FALSE, 'us-gaap'),
        (10, 'EarningsPerShareDiluted', 8, FALSE, 'us-gaap')
    """)

    # Close the connection
    print("Closing connection...")
    conn.close()

    print(f"Test database created at {db_path}")


def test_improved_duckdb_store(db_path):
    """Test the improved DuckDB store with the mock data."""
    print(f"Testing improved DuckDB store with database at {db_path}...")

    # Create the store
    db = ImprovedDuckDBStore(db_path=db_path)

    try:
        # Get database stats
        stats = db.get_database_stats()

        console.print("\n[bold]Database Statistics:[/bold]")
        console.print(f"Companies: {stats.get('companies_count', 'N/A')}")
        console.print(f"Filings: {stats.get('filings_count', 'N/A')}")
        console.print(f"Metrics: {stats.get('metrics_count', 'N/A')}")
        console.print(f"Facts: {stats.get('facts_count', 'N/A')}")
        console.print(
            f"Year Range: {stats.get('min_year', 'N/A')} - {stats.get('max_year', 'N/A')}"
        )

        # Get all companies
        companies = db.get_all_companies()

        console.print("\n[bold]Companies:[/bold]")
        console.print(companies)

        # Get company filings
        ticker = "MSFT"
        filings = db.get_company_filings(ticker)

        console.print(f"\n[bold]Filings for {ticker}:[/bold]")
        console.print(filings)

        # Get filing facts
        filing_id = 1
        facts = db.get_filing_facts(filing_id)

        console.print(f"\n[bold]Facts for Filing ID {filing_id}:[/bold]")
        console.print(facts.head())

        # Query time series
        time_series = db.query_time_series(
            ticker, ["revenue", "net_income"], include_quarterly=True
        )

        console.print(f"\n[bold]Time Series for {ticker}:[/bold]")
        console.print(time_series)

        # Query company comparison
        comparison = db.query_company_comparison(["MSFT", "AAPL"], "revenue")

        console.print("\n[bold]Company Comparison for Revenue:[/bold]")
        console.print(comparison)

        # Query latest metrics
        latest_metrics = db.query_latest_metrics(ticker)

        console.print(f"\n[bold]Latest Metrics for {ticker}:[/bold]")
        console.print(latest_metrics)

        # Run a custom query
        custom_query = """
            SELECT 
                c.ticker,
                m.display_name,
                f.fiscal_year,
                fa.value
            FROM 
                facts fa
            JOIN 
                filings f ON fa.filing_id = f.filing_id
            JOIN 
                companies c ON f.company_id = c.company_id
            JOIN 
                metrics m ON fa.metric_id = m.metric_id
            WHERE 
                m.metric_name IN ('revenue', 'net_income')
                AND f.fiscal_period = 'FY'
            ORDER BY 
                c.ticker, f.fiscal_year, m.metric_name
        """
        custom_result = db.run_custom_query(custom_query)

        console.print("\n[bold]Custom Query Result:[/bold]")
        console.print(custom_result)

        return True

    except Exception as e:
        console.print(f"[red]Error testing improved DuckDB store: {e}[/red]")
        return False

    finally:
        # Close the database connection
        db.close()


def main():
    # Define parameters
    db_path = "data/test_mock_data.duckdb"

    # Create the test database
    create_test_database(db_path)

    # Test the improved DuckDB store
    test_improved_duckdb_store(db_path)


if __name__ == "__main__":
    main()
