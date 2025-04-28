"""
Initialize and test the improved XBRL extractor.

This script initializes a new DuckDB database with the improved schema and tests
the improved XBRL extractor by processing a single filing.
"""

import logging
import os
import sys

import duckdb
from edgar import set_identity
from rich import box
from rich.console import Console
from rich.panel import Panel

from sec_filing_analyzer.data_processing.improved_edgar_xbrl_extractor import (
    ImprovedEdgarXBRLExtractor,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Force stdout to flush immediately
sys.stdout.reconfigure(line_buffering=True)

# Set up console
console = Console()


def initialize_database(db_path):
    """Initialize a new DuckDB database with the improved schema."""
    print(f"Initializing database at {db_path}...")

    # Create the database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Connect to the database
    print("Connecting to database...")
    conn = duckdb.connect(db_path)

    # Initialize the schema
    print("Loading schema...")
    schema_path = "src/sec_filing_analyzer/storage/improved_financial_db_schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    print("Executing schema...")
    conn.execute(schema_sql)

    # Close the connection
    print("Closing connection...")
    conn.close()

    console.print(f"[green]Initialized database at {db_path}[/green]")


def test_extractor(ticker, accession_number, db_path):
    """Test the improved XBRL extractor by processing a single filing."""
    print(f"Testing extractor with {ticker} {accession_number}...")

    # Set edgar identity
    print("Setting edgar identity...")
    set_identity("timothy.yi.wang@gmail.com")  # Using the user's email from the repository

    # Create the extractor
    print("Creating extractor...")
    extractor = ImprovedEdgarXBRLExtractor(db_path=db_path)

    try:
        # Process the filing
        console.print(f"[bold]Processing filing {ticker} {accession_number}...[/bold]")
        result = extractor.process_filing(ticker, accession_number)

        # Display the result
        if "error" in result:
            console.print(
                Panel(
                    f"[red]Error processing filing {ticker} {accession_number}:[/red]\n{result['error']}",
                    title="Error",
                    box=box.ROUNDED,
                )
            )
        else:
            console.print(
                Panel(
                    f"[green]Successfully processed filing {ticker} {accession_number}[/green]\n"
                    f"Filing ID: {result.get('filing_id')}\n"
                    f"Has XBRL: {result.get('has_xbrl')}\n"
                    f"Fiscal Year: {result.get('fiscal_info', {}).get('fiscal_year')}\n"
                    f"Fiscal Period: {result.get('fiscal_info', {}).get('fiscal_period')}",
                    title="Success",
                    box=box.ROUNDED,
                )
            )

        # Query the database to verify the data was stored
        console.print("\n[bold]Verifying data in database...[/bold]")

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
        filing_id = result.get("filing_id")

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
    print("Starting initialization and test...")

    # Define parameters
    db_path = "data/test_improved_financial_data.duckdb"
    ticker = "MSFT"  # Microsoft
    accession_number = "0000789019-22-000010"  # Microsoft 10-K for 2022

    print(f"Using database: {db_path}")
    print(f"Testing with ticker: {ticker}, accession: {accession_number}")

    # Initialize the database
    initialize_database(db_path)

    # Test the extractor
    test_extractor(ticker, accession_number, db_path)


if __name__ == "__main__":
    main()
