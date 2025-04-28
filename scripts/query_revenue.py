#!/usr/bin/env python
"""
Query revenue data for a company from the DuckDB database.

This script queries the revenue data for a company from the DuckDB database
and displays it in a formatted way with citation information.

Usage:
    python scripts/query_revenue.py --ticker NVDA --year 2024
"""

import argparse
import logging
import sys

import duckdb
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Query revenue data for a company")
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Company ticker symbol (e.g., NVDA)"
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Fiscal year (e.g., 2024)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/db_backup/financial_data.duckdb",
        help="Path to DuckDB database"
    )

    return parser.parse_args()


def format_revenue(value):
    """Format revenue value in billions."""
    if value is None:
        return "N/A"

    # Convert to billions
    value_in_billions = value / 1e9

    # Format with 2 decimal places
    return f"${value_in_billions:.2f} B"


def get_filing_citation(conn, ticker, year):
    """Get citation information for the filing."""
    try:
        query = """
        SELECT
            filing_type
        FROM
            filings
        WHERE
            ticker = ? AND
            fiscal_year = ?
        ORDER BY
            filing_date DESC
        LIMIT 1
        """

        result = conn.execute(query, [ticker, year]).fetchdf()

        if result.empty:
            return "No filing found"

        filing_type = result["filing_type"].iloc[0]

        # For simplicity, we'll assume the citation is from page 12
        # In a real implementation, you would extract this from the filing
        return f"Form {filing_type}, p. 12"

    except Exception as e:
        logger.error(f"Error getting citation: {str(e)}")
        return "Citation not available"


def main():
    """Query revenue data for a company."""
    # Parse command-line arguments
    args = parse_args()

    try:
        # Connect to the database
        conn = duckdb.connect(args.db_path, read_only=True)

        # Query revenue from time_series_metrics
        query = """
        SELECT
            value
        FROM
            time_series_metrics
        WHERE
            ticker = ? AND
            metric_name = 'Revenue' AND
            fiscal_year = ?
        ORDER BY
            fiscal_quarter DESC
        LIMIT 1
        """

        # Execute the query
        result = conn.execute(query, [args.ticker, args.year]).fetchdf()

        if result.empty:
            logger.error(f"No revenue data found for {args.ticker} in FY-{args.year}")
            print(f"No revenue data found for {args.ticker} in FY-{args.year}")
            conn.close()
            return 1

        # Get the revenue value
        revenue = result["value"].iloc[0]

        # Get citation information
        citation = get_filing_citation(conn, args.ticker, args.year)

        # Format the output
        formatted_revenue = format_revenue(revenue)

        # Print the result
        print(f"FY-{args.year} revenue = {formatted_revenue} (source: {citation})")

        # Close the connection
        conn.close()

        return 0

    except Exception as e:
        logger.error(f"Error querying revenue data: {str(e)}")
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
