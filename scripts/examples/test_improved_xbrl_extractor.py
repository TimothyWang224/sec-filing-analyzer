"""
Test script for the improved Edgar XBRL extractor.

This script demonstrates how to use the ImprovedEdgarXBRLExtractor class to extract
financial data from SEC filings and store it in a DuckDB database using the improved schema.
"""

import argparse
import logging

from rich import box
from rich.console import Console
from rich.panel import Panel

from sec_filing_analyzer.data_processing.improved_edgar_xbrl_extractor import (
    ImprovedEdgarXBRLExtractor,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up console
console = Console()


def process_filing(ticker: str, accession_number: str, db_path: str):
    """
    Process a single SEC filing.

    Args:
        ticker: Company ticker symbol
        accession_number: SEC accession number
        db_path: Path to the DuckDB database file
    """
    try:
        # Create the extractor
        extractor = ImprovedEdgarXBRLExtractor(db_path=db_path)

        # Process the filing
        logger.info(f"Processing filing {ticker} {accession_number}")
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

        # Close the extractor
        extractor.close()

    except Exception as e:
        logger.error(f"Error processing filing {ticker} {accession_number}: {e}")
        console.print(f"[red]Error: {e}[/red]")


def process_company(
    ticker: str, filing_types: list = None, limit: int = None, db_path: str = None
):
    """
    Process all filings for a company.

    Args:
        ticker: Company ticker symbol
        filing_types: List of filing types to process (e.g., ['10-K', '10-Q'])
        limit: Maximum number of filings to process
        db_path: Path to the DuckDB database file
    """
    try:
        # Create the extractor
        extractor = ImprovedEdgarXBRLExtractor(db_path=db_path)

        # Process the company's filings
        logger.info(f"Processing filings for {ticker}")

        # Default to 10-K and 10-Q filings if not specified
        if filing_types is None:
            filing_types = ["10-K", "10-Q"]

        result = extractor.process_company(ticker, filing_types, limit)

        # Display the result
        if "error" in result:
            console.print(
                Panel(
                    f"[red]Error processing filings for {ticker}:[/red]\n{result['error']}",
                    title="Error",
                    box=box.ROUNDED,
                )
            )
        else:
            console.print(
                Panel(
                    f"[green]Successfully processed filings for {ticker}[/green]\n"
                    f"Processed {len(result.get('results', []))} filings",
                    title="Success",
                    box=box.ROUNDED,
                )
            )

        # Close the extractor
        extractor.close()

    except Exception as e:
        logger.error(f"Error processing company {ticker}: {e}")
        console.print(f"[red]Error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(
        description="Test the improved Edgar XBRL extractor"
    )
    parser.add_argument("--ticker", required=True, help="Company ticker symbol")
    parser.add_argument(
        "--accession", help="SEC accession number (for processing a single filing)"
    )
    parser.add_argument(
        "--filing-type",
        action="append",
        help="Filing type to process (e.g., 10-K, 10-Q)",
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum number of filings to process"
    )
    parser.add_argument(
        "--db",
        default="data/financial_data_new.duckdb",
        help="Path to the DuckDB database file",
    )

    args = parser.parse_args()

    # Process filings
    if args.accession:
        # Process a single filing
        process_filing(
            ticker=args.ticker, accession_number=args.accession, db_path=args.db
        )
    else:
        # Process all filings for a company
        process_company(
            ticker=args.ticker,
            filing_types=args.filing_type,
            limit=args.limit,
            db_path=args.db,
        )


if __name__ == "__main__":
    main()
