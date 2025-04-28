"""
Compare DuckDB Schemas

This script compares the old and new DuckDB schemas by running the same queries
against both databases and comparing the results.
"""

import argparse
import logging

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sec_filing_analyzer.storage.improved_duckdb_store import ImprovedDuckDBStore
from sec_filing_analyzer.storage.optimized_duckdb_store import OptimizedDuckDBStore

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set up console
console = Console()


def compare_database_stats(old_db: OptimizedDuckDBStore, new_db: ImprovedDuckDBStore):
    """Compare database statistics."""
    # Get stats from old database
    old_stats = {}
    try:
        # Count companies
        old_stats["companies_count"] = old_db.conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]

        # Count filings
        old_stats["filings_count"] = old_db.conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]

        # Count facts
        old_stats["facts_count"] = old_db.conn.execute("SELECT COUNT(*) FROM financial_facts").fetchone()[0]

        # Count metrics
        old_stats["metrics_count"] = old_db.conn.execute(
            "SELECT COUNT(DISTINCT metric_name) FROM time_series_metrics"
        ).fetchone()[0]

        # Get year range
        year_range = old_db.conn.execute("SELECT MIN(fiscal_year), MAX(fiscal_year) FROM filings").fetchone()
        old_stats["min_year"] = year_range[0]
        old_stats["max_year"] = year_range[1]
    except Exception as e:
        logger.error(f"Error getting old database stats: {e}")

    # Get stats from new database
    new_stats = new_db.get_database_stats()

    # Display comparison
    console.print("\n[bold]Database Statistics Comparison:[/bold]")

    table = Table(title="Database Statistics", box=box.ROUNDED)
    table.add_column("Statistic", style="cyan")
    table.add_column("Old Schema", style="green")
    table.add_column("New Schema", style="blue")

    table.add_row(
        "Companies",
        str(old_stats.get("companies_count", "N/A")),
        str(new_stats.get("companies_count", "N/A")),
    )
    table.add_row(
        "Filings",
        str(old_stats.get("filings_count", "N/A")),
        str(new_stats.get("filings_count", "N/A")),
    )
    table.add_row(
        "Facts",
        str(old_stats.get("facts_count", "N/A")),
        str(new_stats.get("facts_count", "N/A")),
    )
    table.add_row(
        "Metrics",
        str(old_stats.get("metrics_count", "N/A")),
        str(new_stats.get("metrics_count", "N/A")),
    )
    table.add_row(
        "Year Range",
        f"{old_stats.get('min_year', 'N/A')} - {old_stats.get('max_year', 'N/A')}",
        f"{new_stats.get('min_year', 'N/A')} - {new_stats.get('max_year', 'N/A')}",
    )

    console.print(table)


def compare_companies(old_db: OptimizedDuckDBStore, new_db: ImprovedDuckDBStore):
    """Compare companies in both databases."""
    # Get companies from old database
    old_companies = old_db.conn.execute("SELECT ticker, name FROM companies ORDER BY ticker").fetchdf()

    # Get companies from new database
    new_companies = new_db.get_all_companies()
    if "company_id" in new_companies.columns:
        new_companies = new_companies[["ticker", "name"]]

    # Display comparison
    console.print("\n[bold]Companies Comparison:[/bold]")

    if old_companies.empty and new_companies.empty:
        console.print("[yellow]No companies found in either database.[/yellow]")
        return

    # Compare counts
    console.print(f"Old schema: {len(old_companies)} companies")
    console.print(f"New schema: {len(new_companies)} companies")

    # Find companies in old but not in new
    if not old_companies.empty and not new_companies.empty:
        old_tickers = set(old_companies["ticker"])
        new_tickers = set(new_companies["ticker"])

        only_in_old = old_tickers - new_tickers
        only_in_new = new_tickers - old_tickers

        if only_in_old:
            console.print(f"[yellow]Companies in old schema but not in new: {', '.join(only_in_old)}[/yellow]")

        if only_in_new:
            console.print(f"[yellow]Companies in new schema but not in old: {', '.join(only_in_new)}[/yellow]")

    # Show sample of companies
    limit = min(5, max(len(old_companies), len(new_companies)))

    if not old_companies.empty:
        console.print("\n[bold]Sample companies from old schema:[/bold]")
        console.print(old_companies.head(limit))

    if not new_companies.empty:
        console.print("\n[bold]Sample companies from new schema:[/bold]")
        console.print(new_companies.head(limit))


def compare_filings(old_db: OptimizedDuckDBStore, new_db: ImprovedDuckDBStore, ticker: str):
    """Compare filings for a company in both databases."""
    # Get filings from old database
    try:
        old_filings = old_db.conn.execute(f"""
            SELECT 
                accession_number, filing_type, filing_date, fiscal_year, fiscal_quarter
            FROM filings
            WHERE ticker = '{ticker}'
            ORDER BY filing_date DESC
        """).fetchdf()
    except Exception as e:
        logger.error(f"Error getting filings from old database: {e}")
        old_filings = pd.DataFrame()

    # Get filings from new database
    new_filings = new_db.get_company_filings(ticker)
    if not new_filings.empty and "fiscal_quarter" not in new_filings.columns:
        # Map fiscal_period to fiscal_quarter for comparison
        period_to_quarter = {"Q1": 1, "Q2": 2, "Q3": 3, "FY": 4}
        new_filings["fiscal_quarter"] = new_filings["fiscal_period"].map(period_to_quarter)

    # Display comparison
    console.print(f"\n[bold]Filings Comparison for {ticker}:[/bold]")

    if old_filings.empty and new_filings.empty:
        console.print(f"[yellow]No filings found for {ticker} in either database.[/yellow]")
        return

    # Compare counts
    console.print(f"Old schema: {len(old_filings)} filings")
    console.print(f"New schema: {len(new_filings)} filings")

    # Find filings in old but not in new
    if not old_filings.empty and not new_filings.empty:
        old_accessions = set(old_filings["accession_number"])
        new_accessions = set(new_filings["accession_number"])

        only_in_old = old_accessions - new_accessions
        only_in_new = new_accessions - old_accessions

        if only_in_old:
            console.print(f"[yellow]Filings in old schema but not in new: {len(only_in_old)}[/yellow]")

        if only_in_new:
            console.print(f"[yellow]Filings in new schema but not in old: {len(only_in_new)}[/yellow]")

    # Show sample of filings
    limit = min(5, max(len(old_filings), len(new_filings)))

    if not old_filings.empty:
        console.print("\n[bold]Sample filings from old schema:[/bold]")
        console.print(old_filings.head(limit))

    if not new_filings.empty:
        console.print("\n[bold]Sample filings from new schema:[/bold]")
        console.print(new_filings.head(limit))


def compare_facts(
    old_db: OptimizedDuckDBStore,
    new_db: ImprovedDuckDBStore,
    ticker: str,
    accession_number: str,
):
    """Compare facts for a filing in both databases."""
    # Get filing ID from old database
    try:
        old_filing_id = old_db.conn.execute(f"""
            SELECT id FROM filings
            WHERE ticker = '{ticker}' AND accession_number = '{accession_number}'
        """).fetchone()

        if old_filing_id:
            old_filing_id = old_filing_id[0]
        else:
            console.print(f"[yellow]Filing {accession_number} not found in old database.[/yellow]")
            old_filing_id = None
    except Exception as e:
        logger.error(f"Error getting filing ID from old database: {e}")
        old_filing_id = None

    # Get filing ID from new database
    new_filing_id = new_db.get_filing_id(accession_number)

    if not old_filing_id and not new_filing_id:
        console.print(f"[yellow]Filing {accession_number} not found in either database.[/yellow]")
        return

    # Get facts from old database
    old_facts = pd.DataFrame()
    if old_filing_id:
        try:
            old_facts = old_db.conn.execute(f"""
                SELECT 
                    xbrl_tag, metric_name, value, unit, period_type
                FROM financial_facts
                WHERE filing_id = '{old_filing_id}'
                ORDER BY xbrl_tag
                LIMIT 20
            """).fetchdf()
        except Exception as e:
            logger.error(f"Error getting facts from old database: {e}")

    # Get facts from new database
    new_facts = pd.DataFrame()
    if new_filing_id:
        try:
            new_facts = new_db.get_filing_facts(new_filing_id)
            if not new_facts.empty:
                new_facts = new_facts[["metric_name", "value", "unit_of_measure", "period_type"]]
                new_facts = new_facts.rename(columns={"unit_of_measure": "unit"})
        except Exception as e:
            logger.error(f"Error getting facts from new database: {e}")

    # Display comparison
    console.print(f"\n[bold]Facts Comparison for {ticker} {accession_number}:[/bold]")

    if old_facts.empty and new_facts.empty:
        console.print(f"[yellow]No facts found for {accession_number} in either database.[/yellow]")
        return

    # Compare counts
    if old_filing_id:
        try:
            old_count = old_db.conn.execute(f"""
                SELECT COUNT(*) FROM financial_facts
                WHERE filing_id = '{old_filing_id}'
            """).fetchone()[0]
            console.print(f"Old schema: {old_count} facts")
        except Exception as e:
            logger.error(f"Error counting facts in old database: {e}")

    if new_filing_id:
        try:
            new_count = new_db.conn.execute(f"""
                SELECT COUNT(*) FROM facts
                WHERE filing_id = {new_filing_id}
            """).fetchone()[0]
            console.print(f"New schema: {new_count} facts")
        except Exception as e:
            logger.error(f"Error counting facts in new database: {e}")

    # Show sample of facts
    if not old_facts.empty:
        console.print("\n[bold]Sample facts from old schema:[/bold]")
        console.print(old_facts.head(10))

    if not new_facts.empty:
        console.print("\n[bold]Sample facts from new schema:[/bold]")
        console.print(new_facts.head(10))


def compare_time_series(old_db: OptimizedDuckDBStore, new_db: ImprovedDuckDBStore, ticker: str, metric: str):
    """Compare time series data for a company and metric in both databases."""
    # Get time series data from old database
    try:
        old_data = old_db.conn.execute(f"""
            SELECT 
                fiscal_year, fiscal_quarter, value
            FROM time_series_metrics
            WHERE ticker = '{ticker}' AND metric_name = '{metric}'
            ORDER BY fiscal_year, fiscal_quarter
        """).fetchdf()
    except Exception as e:
        logger.error(f"Error getting time series data from old database: {e}")
        old_data = pd.DataFrame()

    # Get time series data from new database
    new_data = new_db.query_time_series(ticker, [metric], include_quarterly=True)
    if not new_data.empty and metric in new_data.columns:
        new_data = new_data[["fiscal_year", "fiscal_period", metric]]
        new_data = new_data.rename(columns={metric: "value"})
    else:
        new_data = pd.DataFrame()

    # Display comparison
    console.print(f"\n[bold]Time Series Comparison for {ticker} {metric}:[/bold]")

    if old_data.empty and new_data.empty:
        console.print(f"[yellow]No time series data found for {ticker} {metric} in either database.[/yellow]")
        return

    # Compare counts
    console.print(f"Old schema: {len(old_data)} data points")
    console.print(f"New schema: {len(new_data)} data points")

    # Show sample of data
    if not old_data.empty:
        console.print("\n[bold]Sample time series data from old schema:[/bold]")
        console.print(old_data.head(10))

    if not new_data.empty:
        console.print("\n[bold]Sample time series data from new schema:[/bold]")
        console.print(new_data.head(10))


def main():
    parser = argparse.ArgumentParser(description="Compare DuckDB schemas")
    parser.add_argument(
        "--old-db",
        default="data/financial_data.duckdb",
        help="Path to the old DuckDB database",
    )
    parser.add_argument(
        "--new-db",
        default="data/financial_data_new.duckdb",
        help="Path to the new DuckDB database",
    )
    parser.add_argument("--stats", action="store_true", help="Compare database statistics")
    parser.add_argument("--companies", action="store_true", help="Compare companies")
    parser.add_argument("--filings", help="Compare filings for a company (specify ticker)")
    parser.add_argument(
        "--facts",
        nargs=2,
        metavar=("TICKER", "ACCESSION"),
        help="Compare facts for a filing (specify ticker and accession number)",
    )
    parser.add_argument(
        "--time-series",
        nargs=2,
        metavar=("TICKER", "METRIC"),
        help="Compare time series data (specify ticker and metric)",
    )

    args = parser.parse_args()

    # Connect to databases
    old_db = OptimizedDuckDBStore(db_path=args.old_db)
    new_db = ImprovedDuckDBStore(db_path=args.new_db)

    try:
        # Display header
        console.print(
            Panel(
                f"[bold]Comparing DuckDB Schemas[/bold]\nOld schema: {args.old_db}\nNew schema: {args.new_db}",
                box=box.ROUNDED,
            )
        )

        # Execute the requested comparisons
        if args.stats:
            compare_database_stats(old_db, new_db)

        if args.companies:
            compare_companies(old_db, new_db)

        if args.filings:
            compare_filings(old_db, new_db, args.filings)

        if args.facts:
            compare_facts(old_db, new_db, args.facts[0], args.facts[1])

        if args.time_series:
            compare_time_series(old_db, new_db, args.time_series[0], args.time_series[1])

        # If no specific comparison is requested, show a menu
        if not any([args.stats, args.companies, args.filings, args.facts, args.time_series]):
            console.print("\n[bold]Available comparisons:[/bold]")
            console.print("  1. Compare database statistics")
            console.print("  2. Compare companies")
            console.print("  3. Compare filings for a company")
            console.print("  4. Compare facts for a filing")
            console.print("  5. Compare time series data")
            console.print("  0. Exit")

            choice = input("\nEnter your choice (0-5): ")

            if choice == "1":
                compare_database_stats(old_db, new_db)
            elif choice == "2":
                compare_companies(old_db, new_db)
            elif choice == "3":
                ticker = input("Enter company ticker: ")
                compare_filings(old_db, new_db, ticker)
            elif choice == "4":
                ticker = input("Enter company ticker: ")
                accession = input("Enter accession number: ")
                compare_facts(old_db, new_db, ticker, accession)
            elif choice == "5":
                ticker = input("Enter company ticker: ")
                metric = input("Enter metric name: ")
                compare_time_series(old_db, new_db, ticker, metric)
            elif choice == "0":
                console.print("[bold green]Goodbye![/bold green]")
            else:
                console.print("[red]Invalid choice[/red]")

    finally:
        # Close database connections
        old_db.conn.close()
        new_db.close()


if __name__ == "__main__":
    main()
