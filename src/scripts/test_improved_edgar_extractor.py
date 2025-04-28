"""
Test the improved Edgar XBRL extractor.

This script tests the improved Edgar XBRL extractor with a known filing.
"""

import logging
from pathlib import Path

from edgar import set_identity
from rich import box
from rich.console import Console
from rich.panel import Panel

from sec_filing_analyzer.data_processing.improved_edgar_xbrl_extractor import (
    ImprovedEdgarXBRLExtractor,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set up console
console = Console()


def test_improved_edgar_extractor():
    """Test the improved Edgar XBRL extractor with a known filing."""
    try:
        # Create output directory
        output_dir = Path("data/test_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set edgar identity from environment variables
        import os

        edgar_identity = os.getenv("EDGAR_IDENTITY")
        if edgar_identity:
            set_identity(edgar_identity)
            logger.info(f"Set edgar identity to: {edgar_identity}")
        else:
            # Use a default identity
            set_identity("timothy.yi.wang@gmail.com")
            logger.info("Set edgar identity to default value")

        # Create the database
        db_path = "data/test_improved_edgar.duckdb"

        # Initialize the schema
        import duckdb

        from sec_filing_analyzer.storage.improved_duckdb_store import (
            ImprovedDuckDBStore,
        )

        # Create the database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Connect to the database
        conn = duckdb.connect(db_path)

        # Initialize the schema
        schema_path = "src/sec_filing_analyzer/storage/simplified_financial_db_schema.sql"
        with open(schema_path, "r") as f:
            schema_sql = f.read()

        conn.execute(schema_sql)
        conn.close()

        # Create the extractor
        extractor = ImprovedEdgarXBRLExtractor(db_path=db_path)

        # Use a known 10-K filing for Microsoft
        ticker = "MSFT"
        accession_number = "0001564590-22-026876"  # Microsoft's 10-K from July 2022

        # Process the filing
        logger.info(f"Processing filing {ticker} {accession_number}...")
        result = extractor.process_filing(ticker, accession_number)

        # Check if processing was successful
        if "error" in result:
            console.print(
                Panel(
                    f"[red]Error processing filing {ticker} {accession_number}:[/red]\n{result['error']}",
                    title="Error",
                    box=box.ROUNDED,
                )
            )
            return None

        # Display the result
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
        db = ImprovedDuckDBStore(db_path=db_path)

        # Get database stats
        stats = db.get_database_stats()

        console.print("\n[bold]Database Statistics:[/bold]")
        console.print(f"Companies: {stats.get('companies_count', 'N/A')}")
        console.print(f"Filings: {stats.get('filings_count', 'N/A')}")
        console.print(f"Metrics: {stats.get('metrics_count', 'N/A')}")
        console.print(f"Facts: {stats.get('facts_count', 'N/A')}")

        # Get company data
        company = db.get_company(ticker)

        if company:
            console.print("\n[bold]Company Data:[/bold]")
            console.print(f"Company ID: {company.get('company_id')}")
            console.print(f"Ticker: {company.get('ticker')}")
            console.print(f"Name: {company.get('name')}")
            console.print(f"CIK: {company.get('cik')}")

        # Get filing data
        filing = db.get_filing(accession_number)

        if filing:
            console.print("\n[bold]Filing Data:[/bold]")
            console.print(f"Filing ID: {filing.get('filing_id')}")
            console.print(f"Accession Number: {filing.get('accession_number')}")
            console.print(f"Filing Type: {filing.get('filing_type')}")
            console.print(f"Filing Date: {filing.get('filing_date')}")
            console.print(f"Fiscal Year: {filing.get('fiscal_year')}")
            console.print(f"Fiscal Period: {filing.get('fiscal_period')}")

        # Get facts for the filing
        if filing:
            facts = db.get_filing_facts(filing.get("filing_id"))

            console.print(f"\n[bold]Facts for Filing (showing first 5 of {len(facts)}):[/bold]")
            if not facts.empty:
                console.print(facts.head(5))
            else:
                console.print("[yellow]No facts found for this filing.[/yellow]")

        # Close the database connection
        db.close()

        return result

    except Exception as e:
        logger.error(f"Error testing improved Edgar XBRL extractor: {e}")
        console.print(f"[red]Error: {e}[/red]")
        return None


if __name__ == "__main__":
    test_improved_edgar_extractor()
