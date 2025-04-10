"""
Example script for using the improved DuckDB schema.

This script demonstrates how to use the ImprovedDuckDBStore class to interact with the improved DuckDB schema.
"""

import argparse
import logging
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
from pathlib import Path

from sec_filing_analyzer.storage.improved_duckdb_store import ImprovedDuckDBStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up console
console = Console()

def show_database_stats(db: ImprovedDuckDBStore):
    """Show database statistics."""
    stats = db.get_database_stats()
    
    if not stats:
        console.print("[yellow]No database statistics available.[/yellow]")
        return
    
    console.print("\n[bold]Database Statistics:[/bold]")
    console.print(f"Companies: {stats.get('companies_count', 'N/A')}")
    console.print(f"Filings: {stats.get('filings_count', 'N/A')}")
    console.print(f"Metrics: {stats.get('metrics_count', 'N/A')}")
    console.print(f"Facts: {stats.get('facts_count', 'N/A')}")
    console.print(f"Year Range: {stats.get('min_year', 'N/A')} - {stats.get('max_year', 'N/A')}")
    
    if 'filing_types' in stats and stats['filing_types']:
        console.print(f"Filing Types: {', '.join(stats['filing_types'])}")
    
    if 'metric_categories' in stats and stats['metric_categories']:
        console.print(f"Metric Categories: {', '.join(stats['metric_categories'])}")

def show_companies(db: ImprovedDuckDBStore):
    """Show all companies in the database."""
    companies = db.get_all_companies()
    
    if companies.empty:
        console.print("[yellow]No companies found in the database.[/yellow]")
        return
    
    table = Table(title="Companies", box=box.ROUNDED)
    table.add_column("Ticker", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("CIK", style="blue")
    table.add_column("Industry", style="magenta")
    
    for _, row in companies.iterrows():
        table.add_row(
            row['ticker'],
            row['name'] or '',
            row['cik'] or '',
            row['industry'] or ''
        )
    
    console.print(table)

def show_company_filings(db: ImprovedDuckDBStore, ticker: str):
    """Show filings for a company."""
    filings = db.get_company_filings(ticker)
    
    if filings.empty:
        console.print(f"[yellow]No filings found for {ticker}.[/yellow]")
        return
    
    table = Table(title=f"Filings for {ticker}", box=box.ROUNDED)
    table.add_column("Accession Number", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Date", style="blue")
    table.add_column("Fiscal Year", style="magenta")
    table.add_column("Period", style="yellow")
    
    for _, row in filings.iterrows():
        table.add_row(
            row['accession_number'],
            row['filing_type'] or '',
            str(row['filing_date']) if pd.notna(row['filing_date']) else '',
            str(row['fiscal_year']) if pd.notna(row['fiscal_year']) else '',
            row['fiscal_period'] or ''
        )
    
    console.print(table)

def show_metrics(db: ImprovedDuckDBStore, category: str = None):
    """Show metrics in the database."""
    metrics = db.get_all_metrics(category)
    
    if metrics.empty:
        console.print("[yellow]No metrics found in the database.[/yellow]")
        return
    
    title = "Metrics"
    if category:
        title += f" ({category})"
    
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Display Name", style="green")
    table.add_column("Category", style="blue")
    table.add_column("Unit", style="magenta")
    
    for _, row in metrics.iterrows():
        table.add_row(
            row['metric_name'],
            row['display_name'] or '',
            row['category'] or '',
            row['unit_of_measure'] or ''
        )
    
    console.print(table)

def show_time_series(db: ImprovedDuckDBStore, ticker: str, metrics: list = None, start_year: int = None, end_year: int = None):
    """Show time series data for a company."""
    data = db.query_time_series(ticker, metrics, start_year, end_year)
    
    if data.empty:
        console.print(f"[yellow]No time series data found for {ticker}.[/yellow]")
        return
    
    title = f"Time Series Data for {ticker}"
    if metrics:
        title += f" ({', '.join(metrics)})"
    
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Period", style="cyan")
    
    # Add columns for each metric
    metric_columns = [col for col in data.columns if col not in ['period', 'fiscal_year', 'fiscal_period']]
    for metric in metric_columns:
        table.add_column(metric, style="green")
    
    # Add rows
    for _, row in data.iterrows():
        values = [row['period']]
        for metric in metric_columns:
            value = row[metric] if pd.notna(row[metric]) else ''
            values.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        table.add_row(*values)
    
    console.print(table)

def show_company_comparison(db: ImprovedDuckDBStore, tickers: list, metric: str, start_year: int = None, end_year: int = None):
    """Show company comparison data."""
    data = db.query_company_comparison(tickers, metric, start_year, end_year)
    
    if data.empty:
        console.print(f"[yellow]No comparison data found for {metric}.[/yellow]")
        return
    
    title = f"Company Comparison for {metric}"
    
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Period", style="cyan")
    
    # Add columns for each ticker
    for ticker in tickers:
        if ticker in data.columns:
            table.add_column(ticker, style="green")
    
    # Add rows
    for _, row in data.iterrows():
        values = [row['period']]
        for ticker in tickers:
            if ticker in data.columns:
                value = row[ticker] if pd.notna(row[ticker]) else ''
                values.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        table.add_row(*values)
    
    console.print(table)

def show_latest_metrics(db: ImprovedDuckDBStore, ticker: str, category: str = None):
    """Show latest metrics for a company."""
    data = db.query_latest_metrics(ticker, category)
    
    if data.empty:
        console.print(f"[yellow]No latest metrics found for {ticker}.[/yellow]")
        return
    
    title = f"Latest Metrics for {ticker}"
    if category:
        title += f" ({category})"
    
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Category", style="cyan")
    table.add_column("Metric", style="green")
    table.add_column("Value", style="blue")
    table.add_column("Unit", style="magenta")
    
    for _, row in data.iterrows():
        table.add_row(
            row['category'] or '',
            row['display_name'] or '',
            f"{row['value']:.2f}" if pd.notna(row['value']) and isinstance(row['value'], float) else str(row['value']),
            row['unit_of_measure'] or ''
        )
    
    console.print(table)

def run_custom_query(db: ImprovedDuckDBStore, query: str):
    """Run a custom SQL query."""
    try:
        result = db.run_custom_query(query)
        
        if result.empty:
            console.print("[yellow]Query returned no results.[/yellow]")
            return
        
        console.print("\n[bold]Query Results:[/bold]")
        console.print(result)
    except Exception as e:
        console.print(f"[red]Error running query: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description="Example script for using the improved DuckDB schema")
    parser.add_argument("--db", default="data/financial_data_new.duckdb", help="Path to the DuckDB database file")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--companies", action="store_true", help="Show all companies")
    parser.add_argument("--filings", help="Show filings for a company (specify ticker)")
    parser.add_argument("--metrics", nargs="?", const="all", help="Show metrics (optionally specify category)")
    parser.add_argument("--time-series", help="Show time series data for a company (specify ticker)")
    parser.add_argument("--metric", action="append", help="Metric name for time series or comparison (can be used multiple times)")
    parser.add_argument("--compare", nargs="+", help="Compare companies (specify tickers)")
    parser.add_argument("--compare-metric", help="Metric to use for company comparison")
    parser.add_argument("--latest", help="Show latest metrics for a company (specify ticker)")
    parser.add_argument("--category", help="Category filter for metrics or latest metrics")
    parser.add_argument("--start-year", type=int, help="Start year for time series or comparison")
    parser.add_argument("--end-year", type=int, help="End year for time series or comparison")
    parser.add_argument("--query", help="Run a custom SQL query")
    
    args = parser.parse_args()
    
    # Check if database file exists
    db_path = Path(args.db)
    if not db_path.exists():
        console.print(f"[red]Database file not found: {args.db}[/red]")
        console.print("[yellow]Please run the migration script first:[/yellow]")
        console.print("python src/scripts/migrate_duckdb_schema.py")
        return
    
    # Connect to database
    db = ImprovedDuckDBStore(db_path=args.db)
    
    try:
        # Execute the requested command
        if args.stats:
            show_database_stats(db)
        elif args.companies:
            show_companies(db)
        elif args.filings:
            show_company_filings(db, args.filings)
        elif args.metrics:
            category = None if args.metrics == "all" else args.metrics
            show_metrics(db, category)
        elif args.time_series:
            show_time_series(db, args.time_series, args.metric, args.start_year, args.end_year)
        elif args.compare and args.compare_metric:
            show_company_comparison(db, args.compare, args.compare_metric, args.start_year, args.end_year)
        elif args.latest:
            show_latest_metrics(db, args.latest, args.category)
        elif args.query:
            run_custom_query(db, args.query)
        else:
            # If no specific command is given, show a menu
            console.print("\n[bold]Available commands:[/bold]")
            console.print("  1. Show database statistics")
            console.print("  2. Show all companies")
            console.print("  3. Show filings for a company")
            console.print("  4. Show metrics")
            console.print("  5. Show time series data for a company")
            console.print("  6. Compare companies")
            console.print("  7. Show latest metrics for a company")
            console.print("  8. Run a custom SQL query")
            console.print("  0. Exit")
            
            choice = input("\nEnter your choice (0-8): ")
            
            if choice == "1":
                show_database_stats(db)
            elif choice == "2":
                show_companies(db)
            elif choice == "3":
                ticker = input("Enter company ticker: ")
                show_company_filings(db, ticker)
            elif choice == "4":
                category = input("Enter category (leave empty for all): ")
                show_metrics(db, category if category else None)
            elif choice == "5":
                ticker = input("Enter company ticker: ")
                metrics_input = input("Enter metrics (comma-separated, leave empty for all): ")
                metrics = [m.strip() for m in metrics_input.split(",")] if metrics_input else None
                start_year = input("Enter start year (leave empty for all): ")
                end_year = input("Enter end year (leave empty for all): ")
                show_time_series(
                    db, 
                    ticker, 
                    metrics, 
                    int(start_year) if start_year.isdigit() else None,
                    int(end_year) if end_year.isdigit() else None
                )
            elif choice == "6":
                tickers_input = input("Enter tickers to compare (comma-separated): ")
                tickers = [t.strip() for t in tickers_input.split(",")]
                metric = input("Enter metric to compare: ")
                start_year = input("Enter start year (leave empty for all): ")
                end_year = input("Enter end year (leave empty for all): ")
                show_company_comparison(
                    db,
                    tickers,
                    metric,
                    int(start_year) if start_year.isdigit() else None,
                    int(end_year) if end_year.isdigit() else None
                )
            elif choice == "7":
                ticker = input("Enter company ticker: ")
                category = input("Enter category (leave empty for all): ")
                show_latest_metrics(db, ticker, category if category else None)
            elif choice == "8":
                query = input("Enter SQL query: ")
                run_custom_query(db, query)
            elif choice == "0":
                console.print("[bold green]Goodbye![/bold green]")
            else:
                console.print("[red]Invalid choice[/red]")
    
    finally:
        # Close database connection
        db.close()

if __name__ == "__main__":
    main()
