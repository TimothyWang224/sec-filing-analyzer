"""
Create a simplified schema for testing.
"""

import duckdb
import os

def main():
    print("Creating simplified schema...")
    
    # Define database path
    db_path = "data/test_simple_schema.duckdb"
    
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Metrics table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            metric_id INTEGER PRIMARY KEY,
            metric_name VARCHAR UNIQUE NOT NULL,
            display_name VARCHAR,
            category VARCHAR,
            unit_of_measure VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Facts table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            fact_id INTEGER PRIMARY KEY,
            filing_id INTEGER REFERENCES filings(filing_id),
            metric_id INTEGER REFERENCES metrics(metric_id),
            value DOUBLE,
            context_id VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (filing_id, metric_id, context_id)
        )
    """)
    
    # Insert some test data
    print("Inserting test data...")
    
    # Insert a company
    conn.execute("""
        INSERT INTO companies (company_id, ticker, name, cik)
        VALUES (1, 'MSFT', 'Microsoft Corporation', '0000789019')
    """)
    
    # Insert a filing
    conn.execute("""
        INSERT INTO filings (filing_id, company_id, accession_number, filing_type, filing_date, fiscal_year, fiscal_period)
        VALUES (1, 1, '0000789019-22-000010', '10-K', '2022-07-28', 2022, 'FY')
    """)
    
    # Insert some metrics
    conn.execute("""
        INSERT INTO metrics (metric_id, metric_name, display_name, category, unit_of_measure)
        VALUES 
        (1, 'revenue', 'Revenue', 'income_statement', 'USD'),
        (2, 'net_income', 'Net Income', 'income_statement', 'USD'),
        (3, 'total_assets', 'Total Assets', 'balance_sheet', 'USD')
    """)
    
    # Insert some facts
    conn.execute("""
        INSERT INTO facts (fact_id, filing_id, metric_id, value, context_id)
        VALUES 
        (1, 1, 1, 198270000000, 'FY2022'),
        (2, 1, 2, 72738000000, 'FY2022'),
        (3, 1, 3, 364840000000, 'FY2022')
    """)
    
    # Create a view
    print("Creating view...")
    conn.execute("""
        CREATE VIEW IF NOT EXISTS company_metrics AS
        SELECT 
            c.ticker,
            c.name AS company_name,
            f.fiscal_year,
            f.fiscal_period,
            m.metric_name,
            m.display_name,
            m.category,
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
    """)
    
    # Query the view
    print("Querying view...")
    result = conn.execute("SELECT * FROM company_metrics").fetchdf()
    print(result)
    
    # Close the connection
    print("Closing connection...")
    conn.close()
    
    print("Done!")

if __name__ == "__main__":
    main()
