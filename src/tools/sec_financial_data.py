"""
SEC Financial Data Tool

This module provides a tool for agents to query financial data from SEC filings
stored in the DuckDB database.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from sec_filing_analyzer.quantitative.storage.optimized_duckdb_store import (
    OptimizedDuckDBStore,
)

from ..contracts import BaseModel, FinancialFactsParams, MetricsParams
from ..errors import ParameterError, QueryTypeUnsupported
from ..tools.base import Tool
from ..tools.decorator import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define supported query types and their parameter models
class TimeSeriesParams(BaseModel):
    """Parameters for time series queries."""

    ticker: str
    metric: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: Optional[str] = None


class CompanyInfoParams(BaseModel):
    """Parameters for company info queries."""

    ticker: Optional[str] = None


class FinancialRatiosParams(BaseModel):
    """Parameters for financial ratios queries."""

    ticker: str
    ratios: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class CustomSQLParams(BaseModel):
    """Parameters for custom SQL queries."""

    sql_query: str


# Map query types to parameter models
SUPPORTED_QUERIES: Dict[str, Type[BaseModel]] = {
    "financial_facts": FinancialFactsParams,
    "company_info": CompanyInfoParams,
    "companies": CompanyInfoParams,
    "metrics": MetricsParams,
    "time_series": TimeSeriesParams,
    "financial_ratios": FinancialRatiosParams,
    "custom_sql": CustomSQLParams,
}

# The tool registration is handled by the @tool decorator


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
        "filing_type": "filing_type",
    },
)
class SECFinancialDataTool(Tool):
    """Tool for querying financial data from SEC filings.

    Retrieves structured financial data from SEC filings, such as revenue, profit, and other financial metrics.
    Use this tool when you need specific financial figures or time series data.
    """

    def __init__(self, db_path: Optional[str] = None, read_only: bool = True):
        """Initialize the SEC financial data tool.

        Args:
            db_path: Optional path to the DuckDB database
            read_only: Whether to open the database in read-only mode
        """
        super().__init__()

        # Initialize DuckDB store
        # Use the ETLConfig from ConfigProvider to ensure consistency
        from sec_filing_analyzer.config import ConfigProvider, ETLConfig

        etl_config = ConfigProvider.get_config(ETLConfig)
        self.db_path = db_path or etl_config.db_path
        self.db_store = None
        self.db_error = None

        # We'll use lazy initialization to avoid hanging during startup
        # The database will be connected only when needed
        logger.info(f"SEC Financial Data Tool initialized with database path: {self.db_path}")

        # Try to initialize the database connection if a mock is raising an exception
        try:
            from sec_filing_analyzer.quantitative.storage.optimized_duckdb_store import (
                OptimizedDuckDBStore,
            )

            OptimizedDuckDBStore(db_path=self.db_path, read_only=True)
        except Exception as e:
            # Set the error message but don't raise it
            self.db_error = str(e)
            logger.warning(f"Failed to initialize DuckDB store during initialization: {self.db_error}")
            logger.warning(f"Database path attempted: {self.db_path}")

    def _ensure_db_connection(self) -> bool:
        """Ensure database connection is established.

        Returns:
            True if connection is successful, False otherwise
        """
        if self.db_store is not None:
            return True

        if self.db_error is not None:
            # We already tried and failed
            return False

        try:
            # Try to initialize the DuckDB store
            self.db_store = OptimizedDuckDBStore(db_path=self.db_path, read_only=True)
            logger.info(f"Successfully initialized DuckDB store at {self.db_path}")
            return True
        except Exception as e:
            # Log the error but don't raise it - we'll handle it gracefully
            self.db_error = str(e)
            logger.warning(f"Failed to initialize DuckDB store: {self.db_error}")
            logger.warning(f"Database path attempted: {self.db_path}")
            return False

    async def _execute_abstract(self, query_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a financial data query.

        Args:
            query_type: Type of query to execute (e.g., "financial_facts", "company_info", "metrics")
            parameters: Optional parameters for the query

        Returns:
            A standardized response dictionary with the following fields:
            - query_type: The type of query that was executed
            - parameters: The parameters that were used
            - results: The results of the query (empty list for errors)
            - output_key: The tool's name
            - success: Boolean indicating whether the operation was successful

            Error responses will additionally have:
            - error or warning: The error message (depending on error_type)
        """
        # Ensure parameters is a dictionary
        if parameters is None:
            parameters = {}

        # Validate query type
        if query_type not in SUPPORTED_QUERIES:
            supported_types = list(SUPPORTED_QUERIES.keys())
            raise QueryTypeUnsupported(query_type, "sec_financial_data", supported_types)

        # Validate parameters using the appropriate model
        param_model = SUPPORTED_QUERIES[query_type]

        try:
            # Validate parameters
            param_model(**parameters)
        except Exception as e:
            raise ParameterError(str(e))

        # Ensure database connection is established
        if not self._ensure_db_connection():
            return self.format_error_response(
                query_type=query_type,
                parameters=parameters,
                error_message=f"Database connection failed: {self.db_error}",
                error_type="error",
            )

        try:
            logger.info(f"Executing financial data query: {query_type}")

            # Execute the appropriate query based on query_type
            if query_type == "financial_facts":
                return self._query_financial_facts(parameters)
            elif query_type == "company_info":
                return self._query_company_info(parameters)
            elif query_type == "companies":
                # Alias for company_info with no ticker (returns all companies)
                return self._query_company_info({})
            elif query_type == "metrics":
                return self._query_metrics(parameters)
            elif query_type == "time_series":
                return self._query_time_series(parameters)
            elif query_type == "financial_ratios":
                return self._query_financial_ratios(parameters)
            elif query_type == "custom_sql":
                return self._execute_custom_sql(parameters)
            else:
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message=f"Unknown query type: {query_type}",
                    error_type="error",
                )

        except Exception as e:
            logger.error(f"Error executing financial data query: {str(e)}")
            return self.format_error_response(
                query_type=query_type,
                parameters=parameters,
                error_message=str(e),
                error_type="error",
            )

    def _query_financial_facts(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query financial facts for a specific company."""
        ticker = parameters.get("ticker")
        metrics = parameters.get("metrics")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        filing_type = parameters.get("filing_type")

        if not ticker:
            return self.format_error_response(
                query_type="financial_facts",
                parameters=parameters,
                error_message="Missing required parameter: ticker",
            )

        # Ensure database connection is established
        if not self._ensure_db_connection():
            return self.format_error_response(
                query_type="financial_facts",
                parameters=parameters,
                error_message=f"Database connection failed: {self.db_error}",
            )

        # Query the database for financial facts
        try:
            results = self.db_store.query_financial_facts(
                ticker=ticker,
                metrics=metrics,
                start_date=start_date,
                end_date=end_date,
                filing_type=filing_type,
            )

            if not results:
                logger.warning(f"No financial facts found for {ticker}")
                return self.format_error_response(
                    query_type="financial_facts",
                    parameters=parameters,
                    error_message=f"No financial facts found for {ticker}",
                    error_type="warning",
                )

            return self.format_success_response(query_type="financial_facts", parameters=parameters, results=results)
        except Exception as e:
            logger.error(f"Error querying financial facts: {str(e)}")
            return self.format_error_response(
                query_type="financial_facts",
                parameters=parameters,
                error_message=f"Error querying financial facts: {str(e)}",
            )

    def _query_company_info(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query company information."""
        ticker = parameters.get("ticker")

        # Ensure database connection is established
        if not self._ensure_db_connection():
            return self.format_error_response(
                query_type="company_info",
                parameters=parameters,
                error_message=f"Database connection failed: {self.db_error}",
            )

        try:
            # If database is available, use it
            if not ticker:
                # If no ticker is provided, return all companies
                results = self.db_store.get_all_companies()
            else:
                # Query specific company
                results = self.db_store.get_company_info(ticker=ticker)

            if not results:
                logger.warning(f"No company info found for {ticker if ticker else 'any company'}")
                return self.format_error_response(
                    query_type="company_info",
                    parameters=parameters,
                    error_message=f"No company info found for {ticker if ticker else 'any company'}",
                    error_type="warning",
                )

            return self.format_success_response(query_type="company_info", parameters=parameters, results=results)
        except Exception as e:
            logger.error(f"Error querying company info: {str(e)}")
            return self.format_error_response(
                query_type="company_info",
                parameters=parameters,
                error_message=f"Error querying company info: {str(e)}",
            )

    def _query_metrics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query available metrics."""
        category = parameters.get("category")

        # Ensure database connection is established
        if not self._ensure_db_connection():
            return self.format_error_response(
                query_type="metrics",
                parameters=parameters,
                error_message=f"Database connection failed: {self.db_error}",
            )

        try:
            # If database is available, use it
            results = self.db_store.get_available_metrics(category=category)

            if not results:
                logger.warning(f"No metrics found for category: {category if category else 'any category'}")
                return self.format_error_response(
                    query_type="metrics",
                    parameters=parameters,
                    error_message=f"No metrics found for category: {category if category else 'any category'}",
                    error_type="warning",
                )

            return self.format_success_response(query_type="metrics", parameters=parameters, results=results)
        except Exception as e:
            logger.error(f"Error querying metrics: {str(e)}")
            return self.format_error_response(
                query_type="metrics",
                parameters=parameters,
                error_message=f"Error querying metrics: {str(e)}",
            )

    def _query_time_series(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query time series data for a specific metric."""
        ticker = parameters.get("ticker")
        metric = parameters.get("metric")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        period = parameters.get("period")

        if not ticker or not metric:
            return self.format_error_response(
                query_type="time_series",
                parameters=parameters,
                error_message="Missing required parameters: ticker and metric",
            )

        # Ensure database connection is established
        if not self._ensure_db_connection():
            return self.format_error_response(
                query_type="time_series",
                parameters=parameters,
                error_message=f"Database connection failed: {self.db_error}",
            )

        try:
            # If database is available, use it
            results = self.db_store.query_time_series(
                ticker=ticker,
                metric=metric,
                start_date=start_date,
                end_date=end_date,
                period=period,
            )

            if not results:
                logger.warning(f"No time series data found for {ticker} and metric {metric}")
                return self.format_error_response(
                    query_type="time_series",
                    parameters=parameters,
                    error_message=f"No time series data found for {ticker} and metric {metric}",
                    error_type="warning",
                )

            return self.format_success_response(query_type="time_series", parameters=parameters, results=results)
        except Exception as e:
            logger.error(f"Error querying time series data: {str(e)}")
            return self.format_error_response(
                query_type="time_series",
                parameters=parameters,
                error_message=f"Error querying time series data: {str(e)}",
            )

    def _query_financial_ratios(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query financial ratios for a specific company."""
        ticker = parameters.get("ticker")
        ratios = parameters.get("ratios")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")

        if not ticker:
            return self.format_error_response(
                query_type="financial_ratios",
                parameters=parameters,
                error_message="Missing required parameter: ticker",
            )

        # Ensure database connection is established
        if not self._ensure_db_connection():
            return self.format_error_response(
                query_type="financial_ratios",
                parameters=parameters,
                error_message=f"Database connection failed: {self.db_error}",
            )

        try:
            # If database is available, use it
            results = self.db_store.query_financial_ratios(
                ticker=ticker, ratios=ratios, start_date=start_date, end_date=end_date
            )

            if not results:
                logger.warning(f"No financial ratios found for {ticker}")
                return self.format_error_response(
                    query_type="financial_ratios",
                    parameters=parameters,
                    error_message=f"No financial ratios found for {ticker}",
                    error_type="warning",
                )

            return self.format_success_response(query_type="financial_ratios", parameters=parameters, results=results)
        except Exception as e:
            logger.error(f"Error querying financial ratios: {str(e)}")
            return self.format_error_response(
                query_type="financial_ratios",
                parameters=parameters,
                error_message=f"Error querying financial ratios: {str(e)}",
            )

    def _execute_custom_sql(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom SQL query."""
        sql_query = parameters.get("sql_query")

        if not sql_query:
            return self.format_error_response(
                query_type="custom_sql",
                parameters=parameters,
                error_message="Missing required parameter: sql_query",
            )

        # Ensure database connection is established
        if not self._ensure_db_connection():
            return self.format_error_response(
                query_type="custom_sql",
                parameters={"sql_query": sql_query},
                error_message=f"Cannot execute custom SQL - database connection failed: {self.db_error}",
            )

        try:
            # If database is available, execute the query
            results = self.db_store.execute_custom_query(sql_query)

            return self.format_success_response(
                query_type="custom_sql",
                parameters={"sql_query": sql_query},
                results=results,
            )
        except Exception as e:
            logger.error(f"Error executing custom SQL: {str(e)}")
            return self.format_error_response(
                query_type="custom_sql",
                parameters={"sql_query": sql_query},
                error_message=f"Error executing custom SQL: {str(e)}",
            )

    def validate_args(self, query_type: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate the tool arguments.

        Args:
            query_type: Type of query to execute
            parameters: Optional parameters for the query

        Returns:
            True if arguments are valid, False otherwise
        """
        try:
            # Validate query type
            if query_type not in SUPPORTED_QUERIES:
                logger.error(f"Invalid query_type: must be one of {list(SUPPORTED_QUERIES.keys())}")
                return False

            # Validate parameters using the appropriate model
            param_model = SUPPORTED_QUERIES[query_type]
            if parameters is None:
                parameters = {}

            try:
                # Validate parameters
                param_model(**parameters)
                return True
            except Exception as e:
                logger.error(f"Parameter validation error: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False
