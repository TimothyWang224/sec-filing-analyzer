"""
Test script for the Edgar XBRL to DuckDB extractor.

This script demonstrates how to use the EdgarXBRLToDuckDBExtractor to extract
financial data from SEC filings and store it in a DuckDB database.
"""

import argparse
import logging
import os

from dotenv import load_dotenv

from sec_filing_analyzer.data_processing.edgar_xbrl_to_duckdb import (
    EdgarXBRLToDuckDBExtractor,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def test_single_filing(ticker, accession_number, db_path=None):
    """
    Test extracting data from a single filing.

    Args:
        ticker: Company ticker symbol
        accession_number: SEC accession number
        db_path: Path to the DuckDB database file
    """
    try:
        # Create the extractor
        extractor = EdgarXBRLToDuckDBExtractor(db_path=db_path)

        # Process the filing
        logger.info(f"Processing filing {ticker} {accession_number}")
        result = extractor.process_filing(ticker, accession_number)

        # Print the result
        if "error" in result:
            logger.error(f"Error processing filing: {result['error']}")
        else:
            logger.info(f"Successfully processed filing: {result['message']}")

            # Print fiscal information if available
            if "fiscal_info" in result and result["fiscal_info"]:
                fiscal_info = result["fiscal_info"]
                logger.info(f"Fiscal Year: {fiscal_info.get('fiscal_year')}")
                logger.info(f"Fiscal Quarter: {fiscal_info.get('fiscal_quarter')}")
                logger.info(
                    f"Period End Date: {fiscal_info.get('fiscal_period_end_date')}"
                )

        return result

    except Exception as e:
        logger.error(f"Error in test_single_filing: {e}")
        return {"error": str(e)}


def test_company_filings(ticker, filing_types=None, limit=5, db_path=None):
    """
    Test extracting data from multiple filings for a company.

    Args:
        ticker: Company ticker symbol
        filing_types: List of filing types to process (e.g., ['10-K', '10-Q'])
        limit: Maximum number of filings to process
        db_path: Path to the DuckDB database file
    """
    try:
        # Create the extractor
        extractor = EdgarXBRLToDuckDBExtractor(db_path=db_path)

        # Process the company's filings
        logger.info(f"Processing filings for {ticker}")

        # Default to 10-K and 10-Q filings if not specified
        if filing_types is None:
            filing_types = ["10-K", "10-Q"]

        # Process the filings
        result = extractor.process_company_filings(ticker, filing_types, limit)

        # Print the result
        if "error" in result:
            logger.error(f"Error processing filings: {result['error']}")
        else:
            logger.info(f"Successfully processed filings: {result['message']}")

            # Print summary of processed filings
            if "results" in result:
                successful = sum(
                    1
                    for r in result["results"]
                    if "status" in r and r["status"] == "success"
                )
                failed = sum(1 for r in result["results"] if "error" in r)
                logger.info(
                    f"Processed {len(result['results'])} filings: {successful} successful, {failed} failed"
                )

        return result

    except Exception as e:
        logger.error(f"Error in test_company_filings: {e}")
        return {"error": str(e)}


def test_multiple_companies(
    tickers, filing_types=None, limit_per_company=3, db_path=None
):
    """
    Test extracting data from multiple companies.

    Args:
        tickers: List of company ticker symbols
        filing_types: List of filing types to process (e.g., ['10-K', '10-Q'])
        limit_per_company: Maximum number of filings to process per company
        db_path: Path to the DuckDB database file
    """
    try:
        # Create the extractor
        extractor = EdgarXBRLToDuckDBExtractor(db_path=db_path)

        # Process multiple companies
        logger.info(f"Processing filings for {len(tickers)} companies")

        # Default to 10-K and 10-Q filings if not specified
        if filing_types is None:
            filing_types = ["10-K", "10-Q"]

        # Process the companies
        result = extractor.process_multiple_companies(
            tickers, filing_types, limit_per_company
        )

        # Print the result
        if "error" in result:
            logger.error(f"Error processing companies: {result['error']}")
        else:
            logger.info(f"Successfully processed companies: {result['message']}")

            # Print summary of processed companies
            if "results" in result:
                for ticker, company_result in result["results"].items():
                    if "error" in company_result:
                        logger.error(
                            f"Error processing {ticker}: {company_result['error']}"
                        )
                    else:
                        logger.info(
                            f"Successfully processed {ticker}: {company_result['message']}"
                        )

        return result

    except Exception as e:
        logger.error(f"Error in test_multiple_companies: {e}")
        return {"error": str(e)}


def query_database(db_path=None):
    """
    Query the DuckDB database to verify the extracted data.

    Args:
        db_path: Path to the DuckDB database file
    """
    try:
        # Create the extractor (just to get access to the database)
        extractor = EdgarXBRLToDuckDBExtractor(db_path=db_path)

        # Query the database
        logger.info("Querying the database")

        # Get the number of companies
        companies_count = extractor.db.conn.execute(
            "SELECT COUNT(*) FROM companies"
        ).fetchone()[0]
        logger.info(f"Number of companies: {companies_count}")

        # Get the number of filings
        filings_count = extractor.db.conn.execute(
            "SELECT COUNT(*) FROM filings"
        ).fetchone()[0]
        logger.info(f"Number of filings: {filings_count}")

        # Get the number of financial facts
        facts_count = extractor.db.conn.execute(
            "SELECT COUNT(*) FROM financial_facts"
        ).fetchone()[0]
        logger.info(f"Number of financial facts: {facts_count}")

        # Get the number of time series metrics
        metrics_count = extractor.db.conn.execute(
            "SELECT COUNT(*) FROM time_series_metrics"
        ).fetchone()[0]
        logger.info(f"Number of time series metrics: {metrics_count}")

        # Get the top 5 companies by number of filings
        top_companies = extractor.db.conn.execute("""
            SELECT c.ticker, c.name, COUNT(f.id) as filing_count
            FROM companies c
            JOIN filings f ON c.ticker = f.ticker
            GROUP BY c.ticker, c.name
            ORDER BY filing_count DESC
            LIMIT 5
        """).fetchall()

        logger.info("Top 5 companies by number of filings:")
        for ticker, name, count in top_companies:
            logger.info(f"  {ticker} ({name}): {count} filings")

        # Get the top 5 metrics by number of occurrences
        top_metrics = extractor.db.conn.execute("""
            SELECT metric_name, COUNT(*) as count
            FROM time_series_metrics
            GROUP BY metric_name
            ORDER BY count DESC
            LIMIT 5
        """).fetchall()

        logger.info("Top 5 metrics by number of occurrences:")
        for metric, count in top_metrics:
            logger.info(f"  {metric}: {count} occurrences")

    except Exception as e:
        logger.error(f"Error in query_database: {e}")


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(
        description="Test the Edgar XBRL to DuckDB extractor"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "company", "multiple", "query"],
        default="single",
        help="Test mode: single filing, company filings, multiple companies, or query database",
    )
    parser.add_argument("--ticker", help="Company ticker symbol")
    parser.add_argument(
        "--accession", help="SEC accession number (for single filing mode)"
    )
    parser.add_argument(
        "--tickers",
        help="Comma-separated list of ticker symbols (for multiple companies mode)",
    )
    parser.add_argument(
        "--filing-types", help="Comma-separated list of filing types (e.g., 10-K,10-Q)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of filings to process per company",
    )
    parser.add_argument("--db-path", help="Path to the DuckDB database file")

    args = parser.parse_args()

    # Set default database path if not provided
    db_path = args.db_path or "data/improved_financial_data.duckdb"

    # Ensure the database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Parse filing types if provided
    filing_types = None
    if args.filing_types:
        filing_types = args.filing_types.split(",")

    # Run the appropriate test based on the mode
    if args.mode == "single":
        if not args.ticker or not args.accession:
            logger.error(
                "Ticker and accession number are required for single filing mode"
            )
            return

        test_single_filing(args.ticker, args.accession, db_path)

    elif args.mode == "company":
        if not args.ticker:
            logger.error("Ticker is required for company filings mode")
            return

        test_company_filings(args.ticker, filing_types, args.limit, db_path)

    elif args.mode == "multiple":
        if not args.tickers:
            logger.error("Tickers are required for multiple companies mode")
            return

        tickers = args.tickers.split(",")
        test_multiple_companies(tickers, filing_types, args.limit, db_path)

    elif args.mode == "query":
        query_database(db_path)


if __name__ == "__main__":
    # Set edgar identity from environment variables
    edgar_identity = os.getenv("EDGAR_IDENTITY")
    if edgar_identity:
        import edgar

        edgar.set_identity(edgar_identity)
        logger.info(f"Set edgar identity to: {edgar_identity}")
    else:
        logger.warning(
            "EDGAR_IDENTITY environment variable not set. Set it in your .env file."
        )

    main()
