"""
Test script for the SEC Financial Data Tool.

This script tests the functionality of the SECFinancialDataTool by performing
various financial data queries on the DuckDB database.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tools.sec_financial_data import SECFinancialDataTool

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_financial_facts(
    ticker: str,
    metrics: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    filing_type: Optional[str] = None,
):
    """Test querying financial facts."""
    try:
        # Initialize the tool
        tool = SECFinancialDataTool()

        # Execute the query
        logger.info(f"Querying financial facts for company: {ticker}")
        results = await tool.execute(
            query_type="financial_facts",
            parameters={
                "ticker": ticker,
                "metrics": metrics,
                "start_date": start_date,
                "end_date": end_date,
                "filing_type": filing_type,
            },
        )

        # Print results
        print("\n=== Financial Facts ===")
        print(f"Company: {ticker}")
        print(f"Metrics: {metrics}")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Filing Type: {filing_type}")
        print(f"Total Results: {len(results['results'])}")
        print("\nFacts:")

        for i, result in enumerate(results["results"]):
            print(f"\n--- Fact {i + 1} ---")
            print(f"Metric: {result.get('metric_name', 'N/A')}")
            print(f"Value: {result.get('value', 'N/A')}")
            print(f"Period End Date: {result.get('period_end_date', 'N/A')}")
            print(f"Filing Type: {result.get('filing_type', 'N/A')}")
            print(f"Units: {result.get('units', 'N/A')}")

        return results

    except Exception as e:
        logger.error(f"Error testing financial facts query: {str(e)}")
        raise


async def test_company_info(ticker: Optional[str] = None):
    """Test querying company information."""
    try:
        # Initialize the tool
        tool = SECFinancialDataTool()

        # Execute the query
        if ticker:
            logger.info(f"Querying information for company: {ticker}")
        else:
            logger.info("Querying information for all companies")

        results = await tool.execute(query_type="company_info", parameters={"ticker": ticker})

        # Print results
        print("\n=== Company Information ===")
        if ticker:
            print(f"Company: {ticker}")
        else:
            print("All Companies")

        print(f"Total Results: {len(results['results'])}")
        print("\nCompanies:")

        for i, result in enumerate(results["results"]):
            print(f"\n--- Company {i + 1} ---")
            print(f"Ticker: {result.get('ticker', 'N/A')}")
            print(f"Name: {result.get('name', 'N/A')}")
            print(f"CIK: {result.get('cik', 'N/A')}")
            print(f"Sector: {result.get('sector', 'N/A')}")
            print(f"Industry: {result.get('industry', 'N/A')}")

        return results

    except Exception as e:
        logger.error(f"Error testing company info query: {str(e)}")
        raise


async def test_metrics(category: Optional[str] = None):
    """Test querying available metrics."""
    try:
        # Initialize the tool
        tool = SECFinancialDataTool()

        # Execute the query
        if category:
            logger.info(f"Querying metrics for category: {category}")
        else:
            logger.info("Querying all available metrics")

        results = await tool.execute(query_type="metrics", parameters={"category": category})

        # Print results
        print("\n=== Available Metrics ===")
        if category:
            print(f"Category: {category}")
        else:
            print("All Categories")

        print(f"Total Results: {len(results['results'])}")
        print("\nMetrics:")

        for i, result in enumerate(results["results"]):
            print(f"\n--- Metric {i + 1} ---")
            print(f"Name: {result.get('name', 'N/A')}")
            print(f"Label: {result.get('label', 'N/A')}")
            print(f"Category: {result.get('category', 'N/A')}")
            print(f"Description: {result.get('description', 'N/A')}")

        return results

    except Exception as e:
        logger.error(f"Error testing metrics query: {str(e)}")
        raise


async def test_time_series(
    ticker: str,
    metric: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: Optional[str] = None,
):
    """Test querying time series data."""
    try:
        # Initialize the tool
        tool = SECFinancialDataTool()

        # Execute the query
        logger.info(f"Querying time series for {ticker}, metric: {metric}")
        results = await tool.execute(
            query_type="time_series",
            parameters={
                "ticker": ticker,
                "metric": metric,
                "start_date": start_date,
                "end_date": end_date,
                "period": period,
            },
        )

        # Print results
        print("\n=== Time Series Data ===")
        print(f"Company: {ticker}")
        print(f"Metric: {metric}")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Period: {period}")
        print(f"Total Results: {len(results['results'])}")
        print("\nTime Series:")

        for i, result in enumerate(results["results"]):
            print(f"\n--- Data Point {i + 1} ---")
            print(f"Date: {result.get('date', 'N/A')}")
            print(f"Value: {result.get('value', 'N/A')}")
            print(f"Period: {result.get('period', 'N/A')}")

        return results

    except Exception as e:
        logger.error(f"Error testing time series query: {str(e)}")
        raise


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test the SEC Financial Data Tool")
    parser.add_argument(
        "--query_type",
        type=str,
        required=True,
        choices=["financial_facts", "company_info", "metrics", "time_series"],
        help="Type of query to execute",
    )
    parser.add_argument("--ticker", type=str, help="Company ticker symbol")
    parser.add_argument("--metrics", type=str, nargs="*", help="List of metrics to query")
    parser.add_argument("--metric", type=str, help="Single metric for time series query")
    parser.add_argument("--start_date", type=str, help="Start date for query (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date for query (YYYY-MM-DD)")
    parser.add_argument("--filing_type", type=str, help="Filing type to filter by")
    parser.add_argument("--category", type=str, help="Metric category to filter by")
    parser.add_argument("--period", type=str, help="Period for time series query (e.g., 'annual', 'quarterly')")

    args = parser.parse_args()

    # Run the appropriate test based on query_type
    if args.query_type == "financial_facts":
        if not args.ticker:
            parser.error("--ticker is required for financial_facts query")
        asyncio.run(
            test_financial_facts(
                ticker=args.ticker,
                metrics=args.metrics,
                start_date=args.start_date,
                end_date=args.end_date,
                filing_type=args.filing_type,
            )
        )
    elif args.query_type == "company_info":
        asyncio.run(test_company_info(ticker=args.ticker))
    elif args.query_type == "metrics":
        asyncio.run(test_metrics(category=args.category))
    elif args.query_type == "time_series":
        if not args.ticker or not args.metric:
            parser.error("--ticker and --metric are required for time_series query")
        asyncio.run(
            test_time_series(
                ticker=args.ticker,
                metric=args.metric,
                start_date=args.start_date,
                end_date=args.end_date,
                period=args.period,
            )
        )


if __name__ == "__main__":
    main()
