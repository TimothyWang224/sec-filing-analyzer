"""
Query Financial Data Script

This script provides utilities to query financial data stored in DuckDB.
It can generate reports, compare companies, and analyze financial metrics.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sec_filing_analyzer.storage.financial_data_store import FinancialDataStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def show_database_stats(financial_store: FinancialDataStore) -> None:
    """Show database statistics.

    Args:
        financial_store: Financial data store instance
    """
    stats = financial_store.get_database_stats()

    print("\n=== Database Statistics ===")
    print(f"Companies: {stats.get('company_count', 0)}")
    print(f"Filings: {stats.get('filing_count', 0)}")
    print(f"Financial Facts: {stats.get('fact_count', 0)}")
    print(f"Time Series Metrics: {stats.get('time_series_count', 0)}")
    print(f"Financial Ratios: {stats.get('ratio_count', 0)}")

    print("\nCompanies:")
    for ticker in stats.get("companies", []):
        print(f"  {ticker}")

    print("\nFiling Types:")
    for filing_type in stats.get("filing_types", []):
        print(f"  {filing_type}")

    print(f"\nYear Range: {stats.get('min_year', 'N/A')} - {stats.get('max_year', 'N/A')}")


def show_company_metrics(
    financial_store: FinancialDataStore,
    ticker: str,
    metrics: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    quarterly: bool = False,
) -> None:
    """Show company metrics.

    Args:
        financial_store: Financial data store instance
        ticker: Company ticker symbol
        metrics: List of metrics to show (None for all)
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        quarterly: Whether to include quarterly data
    """
    # Default metrics if none provided
    if not metrics:
        metrics = ["revenue", "operating_income", "net_income", "eps_diluted", "total_assets", "stockholders_equity"]

    # Get metrics
    df = financial_store.get_company_metrics(
        ticker=ticker, metrics=metrics, start_year=start_year, end_year=end_year, quarterly=quarterly
    )

    if df.empty:
        print(f"No metrics found for {ticker}")
        return

    print(f"\n=== Financial Metrics for {ticker} ===")
    print(df.to_string(index=False))

    # Plot metrics
    try:
        # Set up the figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        # Plot each metric
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax = axes[i]
                df.plot(x="period", y=metric, kind="bar", ax=ax, color="skyblue")
                ax.set_title(f"{metric.replace('_', ' ').title()} for {ticker}")
                ax.set_xlabel("Period")
                ax.set_ylabel("Value")
                ax.grid(True, linestyle="--", alpha=0.7)

                # Add value labels
                for j, v in enumerate(df[metric]):
                    if pd.notna(v):
                        ax.text(j, v, f"{v:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(f"{ticker}_metrics.png")
        print(f"\nPlot saved as {ticker}_metrics.png")
    except Exception as e:
        logger.error(f"Error plotting metrics: {e}")


def compare_companies(
    financial_store: FinancialDataStore,
    tickers: List[str],
    metric: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    quarterly: bool = False,
) -> None:
    """Compare companies.

    Args:
        financial_store: Financial data store instance
        tickers: List of company tickers
        metric: Metric to compare
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        quarterly: Whether to include quarterly data
    """
    # Get comparison data
    df = financial_store.compare_companies(
        tickers=tickers, metric=metric, start_year=start_year, end_year=end_year, quarterly=quarterly
    )

    if df.empty:
        print(f"No data found for comparison")
        return

    print(f"\n=== Comparison of {metric.replace('_', ' ').title()} ===")
    print(df.to_string(index=False))

    # Plot comparison
    try:
        plt.figure(figsize=(12, 6))

        # Melt the dataframe for easier plotting
        plot_df = df.melt(
            id_vars=["period", "fiscal_year", "fiscal_quarter"],
            value_vars=tickers,
            var_name="Company",
            value_name="Value",
        )

        # Create the plot
        sns.barplot(x="period", y="Value", hue="Company", data=plot_df)

        plt.title(f"Comparison of {metric.replace('_', ' ').title()}")
        plt.xlabel("Period")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Company")

        plt.tight_layout()
        plt.savefig(f"comparison_{metric}.png")
        print(f"\nPlot saved as comparison_{metric}.png")
    except Exception as e:
        logger.error(f"Error plotting comparison: {e}")


def show_financial_ratios(
    financial_store: FinancialDataStore,
    ticker: str,
    ratios: Optional[List[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    quarterly: bool = False,
) -> None:
    """Show financial ratios.

    Args:
        financial_store: Financial data store instance
        ticker: Company ticker symbol
        ratios: List of ratios to show (None for all)
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        quarterly: Whether to include quarterly data
    """
    # Default ratios if none provided
    if not ratios:
        ratios = [
            "gross_margin",
            "operating_margin",
            "net_margin",
            "current_ratio",
            "debt_to_equity",
            "return_on_equity",
        ]

    # Get ratios
    df = financial_store.get_financial_ratios(
        ticker=ticker, ratios=ratios, start_year=start_year, end_year=end_year, quarterly=quarterly
    )

    if df.empty:
        print(f"No ratios found for {ticker}")
        return

    print(f"\n=== Financial Ratios for {ticker} ===")
    print(df.to_string(index=False))

    # Plot ratios
    try:
        # Set up the figure
        fig, axes = plt.subplots(len(ratios), 1, figsize=(12, 4 * len(ratios)))
        if len(ratios) == 1:
            axes = [axes]

        # Plot each ratio
        for i, ratio in enumerate(ratios):
            if ratio in df.columns:
                ax = axes[i]
                df.plot(x="period", y=ratio, kind="bar", ax=ax, color="lightgreen")
                ax.set_title(f"{ratio.replace('_', ' ').title()} for {ticker}")
                ax.set_xlabel("Period")
                ax.set_ylabel("Value")
                ax.grid(True, linestyle="--", alpha=0.7)

                # Add value labels
                for j, v in enumerate(df[ratio]):
                    if pd.notna(v):
                        ax.text(j, v, f"{v:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(f"{ticker}_ratios.png")
        print(f"\nPlot saved as {ticker}_ratios.png")
    except Exception as e:
        logger.error(f"Error plotting ratios: {e}")


def show_filing_info(
    financial_store: FinancialDataStore,
    ticker: Optional[str] = None,
    filing_type: Optional[str] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> None:
    """Show filing information.

    Args:
        financial_store: Financial data store instance
        ticker: Company ticker symbol (None for all)
        filing_type: Filing type (None for all)
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
    """
    # Get filing info
    df = financial_store.get_filing_info(
        ticker=ticker, filing_type=filing_type, start_year=start_year, end_year=end_year
    )

    if df.empty:
        print(f"No filings found")
        return

    print(f"\n=== Filing Information ===")
    print(df.to_string(index=False))


def run_custom_query(financial_store: FinancialDataStore, query: str) -> None:
    """Run a custom SQL query.

    Args:
        financial_store: Financial data store instance
        query: SQL query
    """
    # Run query
    df = financial_store.run_custom_query(query)

    if df.empty:
        print(f"No results found")
        return

    print(f"\n=== Query Results ===")
    print(df.to_string(index=False))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Query financial data from DuckDB")
    parser.add_argument("--db-path", type=str, default="data/financial_data.duckdb", help="Path to DuckDB database")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--ticker", type=str, help="Company ticker symbol")
    parser.add_argument("--metrics", type=str, help="Comma-separated list of metrics to show")
    parser.add_argument("--ratios", type=str, help="Comma-separated list of ratios to show")
    parser.add_argument("--compare", type=str, help="Comma-separated list of tickers to compare")
    parser.add_argument("--metric", type=str, help="Metric to compare")
    parser.add_argument("--filing-type", type=str, help="Filing type (10-K, 10-Q, etc.)")
    parser.add_argument("--start-year", type=int, help="Start year (inclusive)")
    parser.add_argument("--end-year", type=int, help="End year (inclusive)")
    parser.add_argument("--quarterly", action="store_true", help="Include quarterly data")
    parser.add_argument("--filings", action="store_true", help="Show filing information")
    parser.add_argument("--query", type=str, help="Run a custom SQL query")

    args = parser.parse_args()

    # Initialize financial data store
    financial_store = FinancialDataStore(db_path=args.db_path)

    # Show database statistics
    if args.stats:
        show_database_stats(financial_store)

    # Show company metrics
    if args.ticker and not args.compare and not args.filings and not args.ratios:
        metrics = args.metrics.split(",") if args.metrics else None
        show_company_metrics(
            financial_store=financial_store,
            ticker=args.ticker,
            metrics=metrics,
            start_year=args.start_year,
            end_year=args.end_year,
            quarterly=args.quarterly,
        )

    # Show financial ratios
    if args.ticker and args.ratios:
        ratios = args.ratios.split(",")
        show_financial_ratios(
            financial_store=financial_store,
            ticker=args.ticker,
            ratios=ratios,
            start_year=args.start_year,
            end_year=args.end_year,
            quarterly=args.quarterly,
        )

    # Compare companies
    if args.compare and args.metric:
        tickers = args.compare.split(",")
        compare_companies(
            financial_store=financial_store,
            tickers=tickers,
            metric=args.metric,
            start_year=args.start_year,
            end_year=args.end_year,
            quarterly=args.quarterly,
        )

    # Show filing information
    if args.filings:
        show_filing_info(
            financial_store=financial_store,
            ticker=args.ticker,
            filing_type=args.filing_type,
            start_year=args.start_year,
            end_year=args.end_year,
        )

    # Run custom query
    if args.query:
        run_custom_query(financial_store=financial_store, query=args.query)


if __name__ == "__main__":
    main()
