"""
Test script for the SEC Graph Query Tool.

This script tests the functionality of the SECGraphQueryTool by performing
various graph queries on the SEC filing database.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sec_filing_analyzer.config import StorageConfig
from src.tools.sec_graph_query import SECGraphQueryTool

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_company_filings(ticker: str, filing_types: Optional[List[str]] = None, limit: int = 10):
    """Test querying company filings."""
    try:
        # Initialize the tool
        tool = SECGraphQueryTool()

        # Execute the query
        logger.info(f"Querying filings for company: {ticker}")
        results = await tool.execute(
            query_type="company_filings", parameters={"ticker": ticker, "filing_types": filing_types, "limit": limit}
        )

        # Print results
        print("\n=== Company Filings ===")
        print(f"Company: {ticker}")
        print(f"Filing Types: {filing_types}")
        print(f"Total Results: {len(results['results'])}")
        print("\nFilings:")

        for i, result in enumerate(results["results"]):
            print(f"\n--- Filing {i + 1} ---")
            print(f"Type: {result.get('filing_type', 'N/A')}")
            print(f"Date: {result.get('filing_date', 'N/A')}")
            print(f"Accession Number: {result.get('accession_number', 'N/A')}")
            print(f"Fiscal Year: {result.get('fiscal_year', 'N/A')}")
            print(f"Fiscal Period: {result.get('fiscal_period', 'N/A')}")

        return results

    except Exception as e:
        logger.error(f"Error testing company filings query: {str(e)}")
        raise


async def test_filing_sections(accession_number: str, section_types: Optional[List[str]] = None, limit: int = 50):
    """Test querying filing sections."""
    try:
        # Initialize the tool
        tool = SECGraphQueryTool()

        # Execute the query
        logger.info(f"Querying sections for filing: {accession_number}")
        results = await tool.execute(
            query_type="filing_sections",
            parameters={"accession_number": accession_number, "section_types": section_types, "limit": limit},
        )

        # Print results
        print("\n=== Filing Sections ===")
        print(f"Accession Number: {accession_number}")
        print(f"Section Types: {section_types}")
        print(f"Total Results: {len(results['results'])}")
        print("\nSections:")

        for i, result in enumerate(results["results"]):
            print(f"\n--- Section {i + 1} ---")
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Type: {result.get('section_type', 'N/A')}")
            print(f"Order: {result.get('order', 'N/A')}")

        return results

    except Exception as e:
        logger.error(f"Error testing filing sections query: {str(e)}")
        raise


async def test_related_companies(ticker: str, limit: int = 10):
    """Test querying related companies."""
    try:
        # Initialize the tool
        tool = SECGraphQueryTool()

        # Execute the query
        logger.info(f"Querying companies related to: {ticker}")
        results = await tool.execute(query_type="related_companies", parameters={"ticker": ticker, "limit": limit})

        # Print results
        print("\n=== Related Companies ===")
        print(f"Company: {ticker}")
        print(f"Total Results: {len(results['results'])}")
        print("\nRelated Companies:")

        for i, result in enumerate(results["results"]):
            print(f"\n--- Company {i + 1} ---")
            print(f"Ticker: {result.get('ticker', 'N/A')}")
            print(f"Name: {result.get('name', 'N/A')}")
            print(f"Mention Count: {result.get('mention_count', 'N/A')}")

        return results

    except Exception as e:
        logger.error(f"Error testing related companies query: {str(e)}")
        raise


async def test_section_types():
    """Test querying available section types."""
    try:
        # Initialize the tool
        tool = SECGraphQueryTool()

        # Execute the query
        logger.info("Querying available section types")
        results = await tool.execute(query_type="section_types", parameters={})

        # Print results
        print("\n=== Section Types ===")
        print(f"Total Results: {len(results['results'])}")
        print("\nSection Types:")

        for i, result in enumerate(results["results"]):
            print(f"{result.get('section_type', 'N/A')}: {result.get('count', 'N/A')} occurrences")

        return results

    except Exception as e:
        logger.error(f"Error testing section types query: {str(e)}")
        raise


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test the SEC Graph Query Tool")
    parser.add_argument(
        "--query_type",
        type=str,
        required=True,
        choices=["company_filings", "filing_sections", "related_companies", "section_types"],
        help="Type of query to execute",
    )
    parser.add_argument(
        "--ticker", type=str, help="Company ticker symbol (required for company_filings and related_companies)"
    )
    parser.add_argument("--accession_number", type=str, help="Filing accession number (required for filing_sections)")
    parser.add_argument("--filing_types", type=str, nargs="*", help="List of filing types to filter by")
    parser.add_argument("--section_types", type=str, nargs="*", help="List of section types to filter by")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to return")

    args = parser.parse_args()

    # Run the appropriate test based on query_type
    if args.query_type == "company_filings":
        if not args.ticker:
            parser.error("--ticker is required for company_filings query")
        asyncio.run(test_company_filings(ticker=args.ticker, filing_types=args.filing_types, limit=args.limit))
    elif args.query_type == "filing_sections":
        if not args.accession_number:
            parser.error("--accession_number is required for filing_sections query")
        asyncio.run(
            test_filing_sections(
                accession_number=args.accession_number, section_types=args.section_types, limit=args.limit
            )
        )
    elif args.query_type == "related_companies":
        if not args.ticker:
            parser.error("--ticker is required for related_companies query")
        asyncio.run(test_related_companies(ticker=args.ticker, limit=args.limit))
    elif args.query_type == "section_types":
        asyncio.run(test_section_types())


if __name__ == "__main__":
    main()
