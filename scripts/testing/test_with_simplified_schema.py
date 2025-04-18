"""
Test the improved XBRL extractor with a simplified schema.
"""

import os
import logging
import duckdb
from edgar import set_identity
from rich.console import Console
from rich.panel import Panel
from rich import box

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up console
console = Console()

def create_simplified_schema(db_path):
    """Create a simplified schema for testing."""
    print(f"Creating simplified schema at {db_path}...")
    
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
    
    # Insert some standard metrics
    print("Inserting standard metrics...")
    conn.execute("""
        INSERT OR IGNORE INTO metrics (metric_id, metric_name, display_name, description, category, unit_of_measure) VALUES
        (1, 'revenue', 'Revenue', 'Total revenue from operations', 'income_statement', 'USD'),
        (2, 'net_income', 'Net Income', 'Bottom-line earnings after all expenses', 'income_statement', 'USD'),
        (3, 'total_assets', 'Total Assets', 'Sum of all assets', 'balance_sheet', 'USD')
    """)
    
    # Close the connection
    print("Closing connection...")
    conn.close()
    
    print(f"Simplified schema created at {db_path}")

def test_extractor(ticker, accession_number, db_path):
    """Test the improved XBRL extractor."""
    from sec_filing_analyzer.data_processing.improved_edgar_xbrl_extractor import ImprovedEdgarXBRLExtractor
    
    # Set edgar identity
    print("Setting edgar identity...")
    set_identity("timothy.yi.wang@gmail.com")
    
    # Create the extractor
    print(f"Creating extractor with database at {db_path}...")
    extractor = ImprovedEdgarXBRLExtractor(db_path=db_path)
    
    try:
        # Process the filing
        print(f"Processing filing {ticker} {accession_number}...")
        result = extractor.process_filing(ticker, accession_number)
        
        # Display the result
        if "error" in result:
            console.print(Panel(
                f"[red]Error processing filing {ticker} {accession_number}:[/red]\n{result['error']}",
                title="Error",
                box=box.ROUNDED
            ))
        else:
            console.print(Panel(
                f"[green]Successfully processed filing {ticker} {accession_number}[/green]\n"
                f"Filing ID: {result.get('filing_id')}\n"
                f"Has XBRL: {result.get('has_xbrl')}\n"
                f"Fiscal Year: {result.get('fiscal_info', {}).get('fiscal_year')}\n"
                f"Fiscal Period: {result.get('fiscal_info', {}).get('fiscal_period')}",
                title="Success",
                box=box.ROUNDED
            ))
        
        # Query the database to verify the data was stored
        print("\nVerifying data in database...")
        
        # Get database stats
        stats = extractor.db.get_database_stats()
        
        console.print("\n[bold]Database Statistics:[/bold]")
        console.print(f"Companies: {stats.get('companies_count', 'N/A')}")
        console.print(f"Filings: {stats.get('filings_count', 'N/A')}")
        console.print(f"Metrics: {stats.get('metrics_count', 'N/A')}")
        console.print(f"Facts: {stats.get('facts_count', 'N/A')}")
        
        # Get company data
        company = extractor.db.get_company(ticker)
        
        if company:
            console.print("\n[bold]Company Data:[/bold]")
            console.print(f"Company ID: {company.get('company_id')}")
            console.print(f"Ticker: {company.get('ticker')}")
            console.print(f"Name: {company.get('name')}")
            console.print(f"CIK: {company.get('cik')}")
        
        # Get filing data
        filing_id = result.get('filing_id')
        
        if filing_id:
            # Get facts for the filing
            facts = extractor.db.get_filing_facts(filing_id)
            
            console.print(f"\n[bold]Facts for Filing (showing first 5 of {len(facts)}):[/bold]")
            if not facts.empty:
                console.print(facts.head(5))
            else:
                console.print("[yellow]No facts found for this filing.[/yellow]")
        
        return result
    
    finally:
        # Close the extractor
        extractor.close()

def main():
    # Define parameters
    db_path = "data/test_improved_xbrl.duckdb"
    ticker = "MSFT"  # Microsoft
    accession_number = "0000789019-22-000010"  # Microsoft 10-K for 2022
    
    # Create the simplified schema
    create_simplified_schema(db_path)
    
    # Test the extractor
    test_extractor(ticker, accession_number, db_path)

if __name__ == "__main__":
    main()
