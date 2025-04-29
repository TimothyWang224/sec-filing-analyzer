#!/usr/bin/env python
"""
Extension methods for OptimizedDuckDBStore to support the Streamlit demo.
"""

import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import the OptimizedDuckDBStore class
from sec_filing_analyzer.quantitative.storage import OptimizedDuckDBStore


# Add the missing methods to the OptimizedDuckDBStore class
def get_available_companies(self) -> List[str]:
    """Get available companies from the database.

    Returns:
        List of company tickers
    """
    try:
        query = """
            SELECT ticker
            FROM companies
            ORDER BY ticker
        """

        # Execute the query
        result = self.conn.execute(query).fetchdf()

        if result.empty:
            return ["NVDA"]  # Default to NVIDIA if no companies found

        # Convert DataFrame to list
        companies = result["ticker"].tolist()

        return companies
    except Exception as e:
        print(f"Error getting available companies: {str(e)}")
        return ["NVDA"]  # Default to NVIDIA if there's an error


def get_available_metrics(self) -> List[str]:
    """Get available metrics from the database.

    Returns:
        List of metric names
    """
    try:
        query = """
            SELECT DISTINCT metric_name
            FROM financial_facts
            ORDER BY metric_name
        """

        # Execute the query
        result = self.conn.execute(query).fetchdf()

        if result.empty:
            return ["Revenue", "NetIncome", "GrossProfit"]  # Default metrics if no metrics found

        # Convert DataFrame to list
        metrics = result["metric_name"].tolist()

        return metrics
    except Exception as e:
        print(f"Error getting available metrics: {str(e)}")
        return ["Revenue", "NetIncome", "GrossProfit"]  # Default metrics if there's an error


def get_available_years(self) -> List[int]:
    """Get available years from the database.

    Returns:
        List of years
    """
    try:
        query = """
            SELECT DISTINCT fiscal_year
            FROM filings
            WHERE fiscal_year IS NOT NULL
            ORDER BY fiscal_year
        """

        # Execute the query
        result = self.conn.execute(query).fetchdf()

        if result.empty:
            return list(range(2020, 2025))  # Default to 2020-2024 if no years found

        # Convert DataFrame to list and ensure they're integers
        years = [int(y) for y in result["fiscal_year"].tolist() if pd.notna(y)]

        # If no valid years found, return default range
        if not years:
            return list(range(2020, 2025))

        return years
    except Exception as e:
        print(f"Error getting available years: {str(e)}")
        return list(range(2020, 2025))  # Default to 2020-2024 if there's an error


def query_financial_facts(self, ticker: str, metrics: List[str], start_date: str, end_date: str) -> List[dict]:
    """Query financial facts for a company.

    Args:
        ticker: Company ticker
        metrics: List of metrics to query
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        List of financial facts
    """
    try:
        query = """
            SELECT
                f.ticker,
                ff.metric_name,
                ff.value,
                ff.period_end_date,
                ff.source
            FROM
                financial_facts ff
            JOIN
                filings f ON ff.filing_id = f.id
            WHERE
                f.ticker = ?
                AND ff.metric_name IN ({})
                AND ff.period_end_date BETWEEN ? AND ?
            ORDER BY
                ff.period_end_date, ff.metric_name
        """.format(", ".join(["?" for _ in metrics]))

        # Prepare parameters
        params = [ticker] + metrics + [start_date, end_date]

        # Execute the query
        result = self.conn.execute(query, params).fetchdf()

        if result.empty:
            return []

        # Convert DataFrame to list of dictionaries
        facts = []
        for _, row in result.iterrows():
            fact = row.to_dict()
            facts.append(fact)

        return facts
    except Exception as e:
        print(f"Error querying financial facts: {str(e)}")
        return []


# Add the methods to the OptimizedDuckDBStore class
OptimizedDuckDBStore.get_available_companies = get_available_companies
OptimizedDuckDBStore.get_available_metrics = get_available_metrics
OptimizedDuckDBStore.get_available_years = get_available_years
OptimizedDuckDBStore.query_financial_facts = query_financial_facts
