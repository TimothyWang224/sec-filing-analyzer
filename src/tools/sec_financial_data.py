"""
SEC Financial Data Tool

This module provides a tool for agents to query financial data from SEC filings
stored in the DuckDB database.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Type
from datetime import datetime

from ..tools.base import Tool
from ..tools.decorator import tool
from ..contracts import ToolInput, FinancialFactsParams, MetricsParams, BaseModel, ToolSpec
from ..errors import ParameterError, QueryTypeUnsupported, StorageUnavailable, DataNotFound
from sec_filing_analyzer.quantitative.storage.optimized_duckdb_store import OptimizedDuckDBStore

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
    "custom_sql": CustomSQLParams
}

# Register tool specification
from .registry import ToolRegistry

# The tool registration is handled by the @tool decorator
# The ToolSpec will be created automatically by the ToolRegistry._register_tool method

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

        try:
            # Try to initialize the DuckDB store
            self.db_store = OptimizedDuckDBStore(db_path=self.db_path, read_only=read_only)
            logger.info(f"Successfully initialized DuckDB store at {self.db_path}")
        except Exception as e:
            # Log the error but don't raise it - we'll handle it gracefully
            self.db_error = str(e)
            logger.warning(f"Failed to initialize DuckDB store: {self.db_error}")
            logger.warning(f"Database path attempted: {self.db_path}")

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

        Raises:
            QueryTypeUnsupported: If the query type is not supported
            ParameterError: If the parameters are invalid
            StorageUnavailable: If the database is unavailable
            DataNotFound: If the requested data is not found
        """
        # Validate query type
        if query_type not in SUPPORTED_QUERIES:
            supported_types = list(SUPPORTED_QUERIES.keys())
            raise QueryTypeUnsupported(query_type, "sec_financial_data", supported_types)

        # Validate parameters using the appropriate model
        param_model = SUPPORTED_QUERIES[query_type]
        if parameters is None:
            parameters = {}

        try:
            # Validate parameters
            param_model(**parameters)
        except Exception as e:
            raise ParameterError(str(e))

        try:
            logger.info(f"Executing financial data query: {query_type}")

            if parameters is None:
                parameters = {}

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

        # Check if database is available
        if self.db_error:
            return {
                "query_type": "financial_facts",
                "parameters": parameters,
                "error": f"Database connection failed: {self.db_error}",
                "results": []
            }

        # Query the database for financial facts
        try:
            results = self.db_store.query_financial_facts(
                ticker=ticker,
                metrics=metrics,
                start_date=start_date,
                end_date=end_date,
                filing_type=filing_type
            )

            if not results:
                logger.warning(f"No financial facts found for {ticker}")
                return {
                    "query_type": "financial_facts",
                    "parameters": parameters,
                    "warning": f"No financial facts found for {ticker}",
                    "results": []
                }

            return {
                "query_type": "financial_facts",
                "parameters": parameters,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error querying financial facts: {str(e)}")
            return {
                "query_type": "financial_facts",
                "parameters": parameters,
                "error": f"Error querying financial facts: {str(e)}",
                "results": []
            }

    def _query_company_info(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query company information."""
        ticker = parameters.get("ticker")

        # Check if database is available
        if self.db_error:
            return {
                "query_type": "company_info",
                "parameters": parameters,
                "error": f"Database connection failed: {self.db_error}",
                "results": []
            }

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
                return {
                    "query_type": "company_info",
                    "parameters": parameters,
                    "warning": f"No company info found for {ticker if ticker else 'any company'}",
                    "results": []
                }

            return {
                "query_type": "company_info",
                "parameters": parameters,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error querying company info: {str(e)}")
            return {
                "query_type": "company_info",
                "parameters": parameters,
                "error": f"Error querying company info: {str(e)}",
                "results": []
            }

    def _query_metrics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query available metrics."""
        category = parameters.get("category")

        # Check if database is available
        if self.db_error:
            return {
                "query_type": "metrics",
                "parameters": parameters,
                "error": f"Database connection failed: {self.db_error}",
                "results": []
            }

        try:
            # If database is available, use it
            results = self.db_store.get_available_metrics(category=category)

            if not results:
                logger.warning(f"No metrics found for category: {category if category else 'any category'}")
                return {
                    "query_type": "metrics",
                    "parameters": parameters,
                    "warning": f"No metrics found for category: {category if category else 'any category'}",
                    "results": []
                }

            return {
                "query_type": "metrics",
                "parameters": parameters,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error querying metrics: {str(e)}")
            return {
                "query_type": "metrics",
                "parameters": parameters,
                "error": f"Error querying metrics: {str(e)}",
                "results": []
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

        # Check if database is available
        if self.db_error:
            return {
                "query_type": "time_series",
                "parameters": parameters,
                "error": f"Database connection failed: {self.db_error}",
                "results": []
            }

        try:
            # If database is available, use it
            results = self.db_store.query_time_series(
                ticker=ticker,
                metric=metric,
                start_date=start_date,
                end_date=end_date,
                period=period
            )

            if not results:
                logger.warning(f"No time series data found for {ticker} and metric {metric}")
                return {
                    "query_type": "time_series",
                    "parameters": parameters,
                    "warning": f"No time series data found for {ticker} and metric {metric}",
                    "results": []
                }

            return {
                "query_type": "time_series",
                "parameters": parameters,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error querying time series data: {str(e)}")
            return {
                "query_type": "time_series",
                "parameters": parameters,
                "error": f"Error querying time series data: {str(e)}",
                "results": []
            }

    def _query_financial_ratios(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query financial ratios for a specific company."""
        ticker = parameters.get("ticker")
        ratios = parameters.get("ratios")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")

        if not ticker:
            return {"error": "Missing required parameter: ticker", "results": []}

        # Check if database is available
        if self.db_error:
            return {
                "query_type": "financial_ratios",
                "parameters": parameters,
                "error": f"Database connection failed: {self.db_error}",
                "results": []
            }

        try:
            # If database is available, use it
            results = self.db_store.query_financial_ratios(
                ticker=ticker,
                ratios=ratios,
                start_date=start_date,
                end_date=end_date
            )

            if not results:
                logger.warning(f"No financial ratios found for {ticker}")
                return {
                    "query_type": "financial_ratios",
                    "parameters": parameters,
                    "warning": f"No financial ratios found for {ticker}",
                    "results": []
                }

            return {
                "query_type": "financial_ratios",
                "parameters": parameters,
                "results": results
            }
        except Exception as e:
            logger.error(f"Error querying financial ratios: {str(e)}")
            return {
                "query_type": "financial_ratios",
                "parameters": parameters,
                "error": f"Error querying financial ratios: {str(e)}",
                "results": []
            }

    def _execute_custom_sql(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom SQL query."""
        sql_query = parameters.get("sql_query")

        if not sql_query:
            return {"error": "Missing required parameter: sql_query", "results": []}

        # Check if database is available
        if self.db_error:
            return {
                "query_type": "custom_sql",
                "parameters": {
                    "sql_query": sql_query
                },
                "error": "Cannot execute custom SQL - database connection failed",
                "results": [],
                "note": "Database connection failed: " + self.db_error
            }

        # If database is available, execute the query
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
