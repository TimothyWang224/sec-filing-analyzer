"""
SEC Financial Data Tool

This module provides a tool for agents to query financial data from SEC filings
stored in the DuckDB database.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

from ..tools.base import Tool
from ..tools.decorator import tool
from sec_filing_analyzer.quantitative.storage.optimized_duckdb_store import OptimizedDuckDBStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool(
    name="sec_financial_data",
    tags=["sec", "financial", "data"],
    compact_description="Query financial metrics and facts from SEC filings",
    db_schema="financial_facts",
    parameter_mappings={
        "ticker": "ticker",
        "metric": "metric_name",
        "start_date": "period_start_date",
        "end_date": "period_end_date",
        "filing_type": "filing_type"
    }
)
class SECFinancialDataTool(Tool):
    """Tool for querying financial data from SEC filings.

    Retrieves structured financial data from SEC filings, such as revenue, profit, and other financial metrics.
    Use this tool when you need specific financial figures or time series data.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the SEC financial data tool.

        Args:
            db_path: Optional path to the DuckDB database
        """
        super().__init__()

        # Initialize DuckDB store
        self.db_path = db_path or "data/financial_data.duckdb"
        self.db_store = OptimizedDuckDBStore(db_path=self.db_path)

    async def _execute(
        self,
        query_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a financial data query.

        Args:
            query_type: Type of query to execute (e.g., "financial_facts", "company_info", "metrics")
            parameters: Optional parameters for the query

        Returns:
            Dictionary containing query results
        """
        try:
            logger.info(f"Executing financial data query: {query_type}")

            if parameters is None:
                parameters = {}

            # Execute the appropriate query based on query_type
            if query_type == "financial_facts":
                return self._query_financial_facts(parameters)
            elif query_type == "company_info":
                return self._query_company_info(parameters)
            elif query_type == "metrics":
                return self._query_metrics(parameters)
            elif query_type == "time_series":
                return self._query_time_series(parameters)
            elif query_type == "financial_ratios":
                return self._query_financial_ratios(parameters)
            elif query_type == "custom_sql":
                return self._execute_custom_sql(parameters)
            else:
                return {
                    "error": f"Unknown query type: {query_type}",
                    "results": []
                }

        except Exception as e:
            logger.error(f"Error executing financial data query: {str(e)}")
            return {
                "error": str(e),
                "query_type": query_type,
                "parameters": parameters,
                "results": []
            }

    def _query_financial_facts(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query financial facts for a specific company."""
        ticker = parameters.get("ticker")
        metrics = parameters.get("metrics")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        filing_type = parameters.get("filing_type")

        if not ticker:
            return {"error": "Missing required parameter: ticker", "results": []}

        # For testing purposes, return mock data
        # In a real implementation, this would query the database
        results = [
            {
                "ticker": ticker,
                "metric_name": "Revenue",
                "value": "$383.29 billion",
                "period_end_date": "2023-09-30",
                "filing_type": filing_type or "10-K",
                "source": "Mock data for testing"
            }
        ]

        return {
            "query_type": "financial_facts",
            "parameters": parameters,
            "results": results
        }

    def _query_company_info(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query company information."""
        ticker = parameters.get("ticker")

        if not ticker:
            # If no ticker is provided, return all companies
            results = self.db_store.get_all_companies()
        else:
            # Query specific company
            results = self.db_store.get_company_info(ticker=ticker)

        return {
            "query_type": "company_info",
            "parameters": parameters,
            "results": results
        }

    def _query_metrics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query available metrics."""
        category = parameters.get("category")

        # Query metrics
        results = self.db_store.get_available_metrics(category=category)

        return {
            "query_type": "metrics",
            "parameters": parameters,
            "results": results
        }

    def _query_time_series(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query time series data for a specific metric."""
        ticker = parameters.get("ticker")
        metric = parameters.get("metric")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        period = parameters.get("period")

        if not ticker or not metric:
            return {"error": "Missing required parameters: ticker and metric", "results": []}

        # Query time series
        results = self.db_store.query_time_series(
            ticker=ticker,
            metric=metric,
            start_date=start_date,
            end_date=end_date,
            period=period
        )

        return {
            "query_type": "time_series",
            "parameters": parameters,
            "results": results
        }

    def _query_financial_ratios(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query financial ratios for a specific company."""
        ticker = parameters.get("ticker")
        ratios = parameters.get("ratios")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")

        if not ticker:
            return {"error": "Missing required parameter: ticker", "results": []}

        # Query financial ratios
        results = self.db_store.query_financial_ratios(
            ticker=ticker,
            ratios=ratios,
            start_date=start_date,
            end_date=end_date
        )

        return {
            "query_type": "financial_ratios",
            "parameters": parameters,
            "results": results
        }

    def _execute_custom_sql(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom SQL query."""
        sql_query = parameters.get("sql_query")

        if not sql_query:
            return {"error": "Missing required parameter: sql_query", "results": []}

        # Execute custom SQL query
        results = self.db_store.execute_custom_query(sql_query)

        return {
            "query_type": "custom_sql",
            "parameters": {
                "sql_query": sql_query
            },
            "results": results
        }

    def validate_args(
        self,
        query_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate the tool arguments.

        Args:
            query_type: Type of query to execute
            parameters: Optional parameters for the query

        Returns:
            True if arguments are valid, False otherwise
        """
        # Validate query_type
        valid_query_types = [
            "financial_facts",
            "company_info",
            "metrics",
            "time_series",
            "financial_ratios",
            "custom_sql"
        ]

        if not query_type or query_type not in valid_query_types:
            logger.error(f"Invalid query_type: must be one of {valid_query_types}")
            return False

        # Validate parameters based on query_type
        if parameters is None:
            parameters = {}

        if query_type == "financial_facts" and "ticker" not in parameters:
            logger.error("Missing required parameter for financial_facts: ticker")
            return False

        if query_type == "time_series" and ("ticker" not in parameters or "metric" not in parameters):
            logger.error("Missing required parameters for time_series: ticker and metric")
            return False

        if query_type == "financial_ratios" and "ticker" not in parameters:
            logger.error("Missing required parameter for financial_ratios: ticker")
            return False

        if query_type == "custom_sql" and "sql_query" not in parameters:
            logger.error("Missing required parameter for custom_sql: sql_query")
            return False

        # Validate date parameters if provided
        for date_param in ["start_date", "end_date"]:
            if date_param in parameters and parameters[date_param]:
                try:
                    datetime.strptime(parameters[date_param], "%Y-%m-%d")
                except ValueError:
                    logger.error(f"Invalid {date_param}: must be in format 'YYYY-MM-DD'")
                    return False

        return True
