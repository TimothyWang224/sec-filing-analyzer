import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from ..contracts import BaseModel, field_validator
from .base import Tool
from .decorator import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define parameter models
class SECDataParams(BaseModel):
    """Parameters for SEC data retrieval."""

    ticker: str
    filing_type: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    sections: Optional[List[str]] = None

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Ticker must be a non-empty string")
        return v

    @field_validator("filing_type")
    @classmethod
    def validate_filing_type(cls, v):
        if v is not None and v not in ["10-K", "10-Q", "8-K"]:
            raise ValueError("Filing type must be one of: 10-K, 10-Q, 8-K")
        return v

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v):
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Date must be in format 'YYYY-MM-DD'")
        return v

    @field_validator("sections")
    @classmethod
    def validate_sections(cls, v):
        if v is not None and not isinstance(v, list):
            raise ValueError("Sections must be a list of strings")
        return v


# Map query types to parameter models
SUPPORTED_QUERIES: Dict[str, Type[BaseModel]] = {"sec_data": SECDataParams}

# The tool registration is handled by the @tool decorator


@tool(
    name="sec_data",
    tags=["sec", "data"],
    compact_description="Retrieve raw SEC filing data by ticker and filing type",
    # Not using schema mappings for this tool since it has a custom parameter structure
)
class SECDataTool(Tool):
    """Tool for retrieving and processing SEC filing data.

    Use this tool to retrieve raw SEC filing data for a specific company and filing type.
    """

    def __init__(self):
        """Initialize the SEC data tool."""
        super().__init__()

    async def _execute_abstract(self, query_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the SEC data retrieval tool.

        Args:
            query_type: Type of query to execute (e.g., "sec_data")
            parameters: Parameters for the query

        Returns:
            A standardized response dictionary with the following fields:
            - query_type: The type of query that was executed
            - parameters: The parameters that were used
            - results: The retrieved SEC filing data
            - output_key: The tool's name
            - success: Boolean indicating whether the operation was successful
            - ticker: The company ticker symbol
            - filing_type: The type of filing retrieved
            - time_period: The time period covered by the data
            - sections: The sections that were retrieved
            - data: The actual filing data

            Error responses will additionally have:
            - error or warning: The error message (depending on error_type)
        """
        # Ensure parameters is a dictionary
        if parameters is None:
            parameters = {}

        try:
            # Validate query type
            if query_type not in SUPPORTED_QUERIES:
                supported_types = list(SUPPORTED_QUERIES.keys())
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message=f"Unsupported query type: {query_type}. Supported types: {supported_types}",
                )

            # Validate parameters using the appropriate model
            param_model = SUPPORTED_QUERIES[query_type]

            try:
                # Validate parameters
                params = param_model(**parameters)
            except Exception as e:
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message=f"Parameter validation error: {str(e)}",
                )

            # Extract parameters
            ticker = params.ticker
            filing_type = params.filing_type or "10-K"
            start_date = params.start_date or "2023-01-01"
            end_date = params.end_date or "2023-12-31"
            sections = params.sections or [
                "Financial Statements",
                "Management Discussion",
            ]

            try:
                # This is a placeholder for the actual SEC data retrieval logic
                # In practice, this would:
                # 1. Connect to SEC API or database
                # 2. Retrieve specified filings
                # 3. Extract relevant sections
                # 4. Process and format data

                # Create mock data for demonstration purposes
                mock_data = {
                    "financial_statements": {
                        "balance_sheet": {
                            "assets": "500M",
                            "liabilities": "300M",
                            "equity": "200M",
                        },
                        "income_statement": {
                            "revenue": "100M",
                            "net_income": "20M",
                            "eps": "2.00",
                        },
                    },
                    "management_discussion": {
                        "key_points": [
                            "Strong revenue growth",
                            "Improved margins",
                            "Market expansion",
                        ],
                        "risks": [
                            "Market competition",
                            "Regulatory changes",
                            "Economic conditions",
                        ],
                    },
                }

                # Create a custom result with additional fields
                result = self.format_success_response(query_type=query_type, parameters=parameters, results=mock_data)

                # Add additional fields
                result["ticker"] = ticker
                result["filing_type"] = filing_type
                result["time_period"] = {"start": start_date, "end": end_date}
                result["sections"] = sections
                result["data"] = mock_data

                return result

            except Exception as e:
                logger.error(f"Error retrieving SEC data: {str(e)}")
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message=f"Error retrieving SEC data: {str(e)}",
                )

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return self.format_error_response(
                query_type=query_type,
                parameters=parameters,
                error_message=f"Unexpected error: {str(e)}",
            )

    def validate_args(self, query_type: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate the tool arguments.

        Args:
            query_type: Type of query to execute
            parameters: Parameters for the query

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
