"""
DuckDB Schema Migration Script

This script migrates data from the old DuckDB schema to the new improved schema.
"""

import argparse
import duckdb
import os
import logging
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up console
console = Console()

def migrate_database(old_db_path, new_db_path, schema_path):
    """
    Migrate data from old schema to new schema.
    
    Args:
        old_db_path: Path to the old DuckDB database
        new_db_path: Path to the new DuckDB database
        schema_path: Path to the new schema SQL file
    """
    # Check if old database exists
    if not os.path.exists(old_db_path):
        console.print(f"[red]Error: Old database not found at {old_db_path}[/red]")
        return False
    
    # Check if schema file exists
    if not os.path.exists(schema_path):
        console.print(f"[red]Error: Schema file not found at {schema_path}[/red]")
        return False
    
    # Create new database directory if it doesn't exist
    new_db_dir = os.path.dirname(new_db_path)
    if new_db_dir and not os.path.exists(new_db_dir):
        os.makedirs(new_db_dir)
    
    # Connect to old database
    console.print(f"[bold]Connecting to old database: {old_db_path}[/bold]")
    old_conn = duckdb.connect(old_db_path)
    
    # Connect to new database
    console.print(f"[bold]Creating new database: {new_db_path}[/bold]")
    new_conn = duckdb.connect(new_db_path)
    
    try:
        # Initialize new schema
        console.print("[bold]Initializing new schema...[/bold]")
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        new_conn.execute(schema_sql)
        
        # Migrate companies
        console.print("[bold]Migrating companies...[/bold]")
        migrate_companies(old_conn, new_conn)
        
        # Migrate filings
        console.print("[bold]Migrating filings...[/bold]")
        migrate_filings(old_conn, new_conn)
        
        # Migrate metrics and facts
        console.print("[bold]Migrating metrics and facts...[/bold]")
        migrate_metrics_and_facts(old_conn, new_conn)
        
        console.print("[bold green]Migration completed successfully![/bold green]")
        return True
    
    except Exception as e:
        console.print(f"[red]Error during migration: {e}[/red]")
        return False
    
    finally:
        old_conn.close()
        new_conn.close()

def migrate_companies(old_conn, new_conn):
    """
    Migrate companies from old schema to new schema.
    
    Args:
        old_conn: Connection to old database
        new_conn: Connection to new database
    """
    # Check if old companies table exists
    tables = old_conn.execute("SHOW TABLES").fetchall()
    if not any(table[0] == 'companies' for table in tables):
        console.print("[yellow]Warning: Companies table not found in old database[/yellow]")
        return
    
    # Get companies from old database
    companies = old_conn.execute("SELECT * FROM companies").fetchdf()
    
    if companies.empty:
        console.print("[yellow]No companies found in old database[/yellow]")
        return
    
    # Create company_id column
    companies['company_id'] = range(1, len(companies) + 1)
    
    # Add updated_at column
    companies['updated_at'] = companies['created_at']
    
    # Register DataFrame
    new_conn.register("temp_companies", companies)
    
    # Insert into new database
    new_conn.execute("""
        INSERT INTO companies (
            company_id, ticker, name, cik, sic, sector, industry, exchange, 
            created_at, updated_at
        )
        SELECT 
            company_id, ticker, name, cik, sic, sector, industry, exchange, 
            created_at, updated_at
        FROM temp_companies
    """)
    
    console.print(f"[green]Migrated {len(companies)} companies[/green]")

def migrate_filings(old_conn, new_conn):
    """
    Migrate filings from old schema to new schema.
    
    Args:
        old_conn: Connection to old database
        new_conn: Connection to new database
    """
    # Check if old filings table exists
    tables = old_conn.execute("SHOW TABLES").fetchall()
    if not any(table[0] == 'filings' for table in tables):
        console.print("[yellow]Warning: Filings table not found in old database[/yellow]")
        return
    
    # Get filings from old database
    filings = old_conn.execute("""
        SELECT 
            f.id AS old_id, 
            f.ticker, 
            f.accession_number, 
            f.filing_type, 
            f.filing_date, 
            f.fiscal_year, 
            f.fiscal_quarter, 
            f.fiscal_period_end_date, 
            f.document_url, 
            f.has_xbrl, 
            f.created_at,
            c.company_id
        FROM 
            filings f
        JOIN 
            companies c ON f.ticker = c.ticker
    """).fetchdf()
    
    if filings.empty:
        console.print("[yellow]No filings found in old database[/yellow]")
        return
    
    # Create filing_id column
    filings['filing_id'] = range(1, len(filings) + 1)
    
    # Map fiscal_quarter to fiscal_period
    filings['fiscal_period'] = filings['fiscal_quarter'].apply(
        lambda q: f"Q{q}" if q in [1, 2, 3] else "FY"
    )
    
    # Add updated_at column
    filings['updated_at'] = filings['created_at']
    
    # Register DataFrame
    new_conn.register("temp_filings", filings)
    
    # Insert into new database
    new_conn.execute("""
        INSERT INTO filings (
            filing_id, company_id, accession_number, filing_type, filing_date,
            fiscal_year, fiscal_period, fiscal_period_end_date, document_url,
            has_xbrl, created_at, updated_at
        )
        SELECT 
            filing_id, company_id, accession_number, filing_type, filing_date,
            fiscal_year, fiscal_period, fiscal_period_end_date, document_url,
            has_xbrl, created_at, updated_at
        FROM temp_filings
    """)
    
    # Create mapping from old filing IDs to new filing IDs
    new_conn.execute("""
        CREATE TABLE IF NOT EXISTS temp_filing_id_map (
            old_id VARCHAR,
            filing_id INTEGER
        )
    """)
    
    new_conn.execute("""
        INSERT INTO temp_filing_id_map (old_id, filing_id)
        SELECT old_id, filing_id FROM temp_filings
    """)
    
    console.print(f"[green]Migrated {len(filings)} filings[/green]")

def migrate_metrics_and_facts(old_conn, new_conn):
    """
    Migrate metrics and facts from old schema to new schema.
    
    Args:
        old_conn: Connection to old database
        new_conn: Connection to new database
    """
    # Check if old financial_facts table exists
    tables = old_conn.execute("SHOW TABLES").fetchall()
    if not any(table[0] == 'financial_facts' for table in tables):
        console.print("[yellow]Warning: Financial facts table not found in old database[/yellow]")
        return
    
    # Get unique metric names from financial_facts
    metrics_from_facts = old_conn.execute("""
        SELECT DISTINCT metric_name
        FROM financial_facts
        WHERE metric_name IS NOT NULL
    """).fetchdf()
    
    # Get unique metric names from time_series_metrics if it exists
    metrics_from_time_series = pd.DataFrame()
    if any(table[0] == 'time_series_metrics' for table in tables):
        metrics_from_time_series = old_conn.execute("""
            SELECT DISTINCT metric_name
            FROM time_series_metrics
            WHERE metric_name IS NOT NULL
        """).fetchdf()
    
    # Get unique ratio names from financial_ratios if it exists
    metrics_from_ratios = pd.DataFrame()
    if any(table[0] == 'financial_ratios' for table in tables):
        metrics_from_ratios = old_conn.execute("""
            SELECT DISTINCT ratio_name AS metric_name
            FROM financial_ratios
            WHERE ratio_name IS NOT NULL
        """).fetchdf()
    
    # Combine all metric names
    all_metrics = pd.concat([
        metrics_from_facts, 
        metrics_from_time_series, 
        metrics_from_ratios
    ]).drop_duplicates()
    
    if all_metrics.empty:
        console.print("[yellow]No metrics found in old database[/yellow]")
        return
    
    # Create metric_id column
    all_metrics['metric_id'] = range(1000, 1000 + len(all_metrics))
    
    # Add display_name column (capitalize words and replace underscores with spaces)
    all_metrics['display_name'] = all_metrics['metric_name'].apply(
        lambda x: ' '.join(word.capitalize() for word in x.split('_'))
    )
    
    # Determine category based on metric name
    def determine_category(metric_name):
        income_keywords = ['revenue', 'income', 'profit', 'loss', 'earnings', 'expense', 'margin', 'ebitda', 'eps']
        balance_keywords = ['asset', 'liability', 'equity', 'debt', 'cash', 'receivable', 'payable', 'inventory']
        cash_flow_keywords = ['cash_flow', 'operating_cash', 'investing_cash', 'financing_cash', 'capex']
        ratio_keywords = ['ratio', 'margin', 'return', 'turnover', 'coverage', 'per_share']
        
        metric_lower = metric_name.lower()
        
        if any(keyword in metric_lower for keyword in income_keywords):
            return 'income_statement'
        elif any(keyword in metric_lower for keyword in balance_keywords):
            return 'balance_sheet'
        elif any(keyword in metric_lower for keyword in cash_flow_keywords):
            return 'cash_flow'
        elif any(keyword in metric_lower for keyword in ratio_keywords):
            return 'ratio'
        else:
            return 'other'
    
    all_metrics['category'] = all_metrics['metric_name'].apply(determine_category)
    
    # Determine unit of measure based on category
    def determine_unit(category):
        if category == 'ratio':
            return 'ratio'
        else:
            return 'USD'
    
    all_metrics['unit_of_measure'] = all_metrics['category'].apply(determine_unit)
    
    # Add other required columns
    all_metrics['description'] = all_metrics['display_name']
    all_metrics['is_calculated'] = False
    all_metrics['calculation_formula'] = None
    all_metrics['created_at'] = 'CURRENT_TIMESTAMP'
    all_metrics['updated_at'] = 'CURRENT_TIMESTAMP'
    
    # Register DataFrame
    new_conn.register("temp_metrics", all_metrics)
    
    # Insert into new database, but don't overwrite existing standard metrics
    new_conn.execute("""
        INSERT INTO metrics (
            metric_id, metric_name, display_name, description, category,
            unit_of_measure, is_calculated, calculation_formula,
            created_at, updated_at
        )
        SELECT 
            tm.metric_id, tm.metric_name, tm.display_name, tm.description, tm.category,
            tm.unit_of_measure, tm.is_calculated, tm.calculation_formula,
            CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
        FROM 
            temp_metrics tm
        WHERE 
            NOT EXISTS (
                SELECT 1 FROM metrics m WHERE m.metric_name = tm.metric_name
            )
    """)
    
    # Create a mapping table for metric names to metric IDs
    new_conn.execute("""
        CREATE TABLE IF NOT EXISTS temp_metric_id_map AS
        SELECT metric_name, metric_id FROM metrics
    """)
    
    console.print(f"[green]Migrated {len(all_metrics)} metrics[/green]")
    
    # Migrate financial facts
    console.print("[bold]Migrating financial facts...[/bold]")
    
    # Get facts from old database
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("[cyan]Migrating facts...", total=100)
        
        # Process in batches to avoid memory issues
        batch_size = 10000
        offset = 0
        total_facts = 0
        
        while True:
            facts = old_conn.execute(f"""
                SELECT 
                    ff.id AS old_id,
                    ff.filing_id AS old_filing_id,
                    ff.xbrl_tag,
                    ff.metric_name,
                    ff.value,
                    ff.unit,
                    ff.period_type,
                    ff.start_date,
                    ff.end_date,
                    ff.segment,
                    ff.context_id,
                    ff.decimals,
                    ff.created_at,
                    tm.filing_id,
                    mm.metric_id
                FROM 
                    financial_facts ff
                JOIN 
                    temp_filing_id_map tm ON ff.filing_id = tm.old_id
                LEFT JOIN
                    temp_metric_id_map mm ON ff.metric_name = mm.metric_name
                ORDER BY ff.id
                LIMIT {batch_size} OFFSET {offset}
            """).fetchdf()
            
            if facts.empty:
                break
            
            # Create fact_id column
            facts['fact_id'] = range(offset + 1, offset + len(facts) + 1)
            
            # Add required columns
            facts['as_reported'] = True
            facts['normalized_value'] = facts['value']
            facts['updated_at'] = facts['created_at']
            
            # Register DataFrame
            new_conn.register("temp_facts", facts)
            
            # Insert into new database
            new_conn.execute("""
                INSERT INTO facts (
                    fact_id, filing_id, metric_id, value, as_reported,
                    normalized_value, period_type, start_date, end_date,
                    context_id, decimals, created_at, updated_at
                )
                SELECT 
                    fact_id, filing_id, metric_id, value, as_reported,
                    normalized_value, period_type, start_date, end_date,
                    context_id, decimals, created_at, updated_at
                FROM 
                    temp_facts
                WHERE
                    metric_id IS NOT NULL
            """)
            
            total_facts += len(facts)
            offset += batch_size
            progress.update(task, completed=min(100, offset * 100 // (offset + batch_size)))
    
    console.print(f"[green]Migrated {total_facts} facts[/green]")
    
    # Migrate XBRL tag mappings if they exist
    if any(table[0] == 'xbrl_tag_mappings' for table in tables):
        console.print("[bold]Migrating XBRL tag mappings...[/bold]")
        
        # Get mappings from old database
        mappings = old_conn.execute("""
            SELECT 
                xtm.xbrl_tag,
                xtm.standard_metric_name AS metric_name,
                xtm.category,
                xtm.description,
                xtm.is_custom,
                xtm.created_at,
                mm.metric_id
            FROM 
                xbrl_tag_mappings xtm
            LEFT JOIN
                temp_metric_id_map mm ON xtm.standard_metric_name = mm.metric_name
        """).fetchdf()
        
        if not mappings.empty:
            # Create mapping_id column
            mappings['mapping_id'] = range(1, len(mappings) + 1)
            
            # Add required columns
            mappings['taxonomy'] = 'us-gaap'
            mappings['taxonomy_version'] = None
            mappings['updated_at'] = mappings['created_at']
            
            # Register DataFrame
            new_conn.register("temp_mappings", mappings)
            
            # Insert into new database
            new_conn.execute("""
                INSERT INTO xbrl_tag_mappings (
                    mapping_id, xbrl_tag, metric_id, is_custom,
                    taxonomy, taxonomy_version, created_at, updated_at
                )
                SELECT 
                    mapping_id, xbrl_tag, metric_id, is_custom,
                    taxonomy, taxonomy_version, created_at, updated_at
                FROM 
                    temp_mappings
                WHERE
                    metric_id IS NOT NULL
                    AND NOT EXISTS (
                        SELECT 1 FROM xbrl_tag_mappings x WHERE x.xbrl_tag = temp_mappings.xbrl_tag
                    )
            """)
            
            console.print(f"[green]Migrated {len(mappings)} XBRL tag mappings[/green]")
    
    # Clean up temporary tables
    new_conn.execute("DROP TABLE IF EXISTS temp_filing_id_map")
    new_conn.execute("DROP TABLE IF EXISTS temp_metric_id_map")

def main():
    parser = argparse.ArgumentParser(description="Migrate DuckDB schema")
    parser.add_argument("--old-db", default="data/financial_data.duckdb", help="Path to the old DuckDB database")
    parser.add_argument("--new-db", default="data/financial_data_new.duckdb", help="Path to the new DuckDB database")
    parser.add_argument("--schema", default="src/sec_filing_analyzer/storage/improved_financial_db_schema.sql", help="Path to the new schema SQL file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the new database if it exists")
    
    args = parser.parse_args()
    
    # Check if new database exists
    if os.path.exists(args.new_db) and not args.overwrite:
        console.print(f"[yellow]Warning: New database already exists at {args.new_db}[/yellow]")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            console.print("[bold]Migration aborted.[/bold]")
            return
    
    # Run migration
    success = migrate_database(args.old_db, args.new_db, args.schema)
    
    if success:
        console.print("\n[bold green]Migration completed successfully![/bold green]")
        console.print(f"New database created at: [cyan]{args.new_db}[/cyan]")
    else:
        console.print("\n[bold red]Migration failed![/bold red]")

if __name__ == "__main__":
    main()
