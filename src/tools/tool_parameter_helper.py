"""
Tool Parameter Helper

This module provides helper functions for generating and validating tool parameters.
"""

import logging
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tool parameter schemas
TOOL_PARAMETER_SCHEMAS = {
    "test_financial_data": {
        "query_type": {
            "type": "string",
            "required": True,
            "enum": ["revenue", "profit", "metrics"],
            "description": "Type of financial data query to execute",
        },
        "parameters": {
            "type": "object",
            "required": True,
            "properties": {
                "ticker": {
                    "type": "string",
                    "required": True,
                    "description": "Company ticker symbol",
                },
                "start_date": {
                    "type": "string",
                    "required": False,
                    "description": "Start date for data retrieval (YYYY-MM-DD)",
                },
                "end_date": {
                    "type": "string",
                    "required": False,
                    "description": "End date for data retrieval (YYYY-MM-DD)",
                },
            },
        },
    },
    "sec_financial_data": {
        "query_type": {
            "type": "string",
            "required": True,
            "enum": [
                "financial_facts",
                "company_info",
                "metrics",
                "time_series",
                "financial_ratios",
                "custom_sql",
            ],
            "description": "Type of financial data query to execute",
        },
        "parameters": {
            "type": "object",
            "required": False,
            "properties": {
                "ticker": {
                    "type": "string",
                    "required_for": [
                        "financial_facts",
                        "time_series",
                        "financial_ratios",
                    ],
                    "description": "Company ticker symbol",
                },
                "metrics": {
                    "type": "array",
                    "required": False,
                    "description": "List of metrics to retrieve",
                },
                "start_date": {
                    "type": "string",
                    "required": False,
                    "description": "Start date for data retrieval (YYYY-MM-DD)",
                },
                "end_date": {
                    "type": "string",
                    "required": False,
                    "description": "End date for data retrieval (YYYY-MM-DD)",
                },
                "filing_type": {
                    "type": "string",
                    "required": False,
                    "enum": ["10-K", "10-Q", "8-K"],
                    "description": "Type of SEC filing",
                },
                "metric": {
                    "type": "string",
                    "required_for": ["time_series"],
                    "description": "Specific metric for time series data",
                },
                "period": {
                    "type": "string",
                    "required": False,
                    "enum": ["annual", "quarterly"],
                    "description": "Time period for data aggregation",
                },
                "ratios": {
                    "type": "array",
                    "required": False,
                    "description": "List of financial ratios to calculate",
                },
                "sql_query": {
                    "type": "string",
                    "required_for": ["custom_sql"],
                    "description": "Custom SQL query to execute",
                },
            },
        },
    },
    "sec_data": {
        "ticker": {
            "type": "string",
            "required": True,
            "description": "Company ticker symbol",
        },
        "filing_type": {
            "type": "string",
            "required": False,
            "enum": ["10-K", "10-Q", "8-K"],
            "description": "Type of SEC filing",
        },
        "start_date": {
            "type": "string",
            "required": False,
            "description": "Start date for filing search (YYYY-MM-DD)",
        },
        "end_date": {
            "type": "string",
            "required": False,
            "description": "End date for filing search (YYYY-MM-DD)",
        },
        "sections": {
            "type": "array",
            "required": False,
            "description": "Specific sections to extract",
        },
    },
    "sec_semantic_search": {
        "query": {
            "type": "string",
            "required": True,
            "description": "Search query text",
        },
        "companies": {
            "type": "array",
            "required": False,
            "description": "List of company tickers to search within",
        },
        "top_k": {
            "type": "integer",
            "required": False,
            "default": 5,
            "description": "Number of results to return",
        },
        "filing_types": {
            "type": "array",
            "required": False,
            "description": "List of filing types to filter by",
        },
        "date_range": {
            "type": "array",
            "required": False,
            "description": "Date range for search [start_date, end_date]",
        },
        "sections": {
            "type": "array",
            "required": False,
            "description": "List of document sections to filter by",
        },
    },
    "sec_graph_query": {
        "query_type": {
            "type": "string",
            "required": True,
            "enum": [
                "company_filings",
                "filing_sections",
                "related_companies",
                "filing_timeline",
                "section_types",
                "custom_cypher",
            ],
            "description": "Type of graph query to execute",
        },
        "parameters": {
            "type": "object",
            "required": False,
            "properties": {
                "ticker": {
                    "type": "string",
                    "required_for": [
                        "company_filings",
                        "filing_sections",
                        "related_companies",
                    ],
                    "description": "Company ticker symbol",
                },
                "filing_type": {
                    "type": "string",
                    "required": False,
                    "enum": ["10-K", "10-Q", "8-K"],
                    "description": "Type of SEC filing",
                },
                "start_date": {
                    "type": "string",
                    "required": False,
                    "description": "Start date for query (YYYY-MM-DD)",
                },
                "end_date": {
                    "type": "string",
                    "required": False,
                    "description": "End date for query (YYYY-MM-DD)",
                },
                "section_type": {
                    "type": "string",
                    "required_for": ["filing_sections"],
                    "description": "Type of section to retrieve",
                },
                "cypher_query": {
                    "type": "string",
                    "required_for": ["custom_cypher"],
                    "description": "Custom Cypher query to execute",
                },
            },
        },
    },
}


def get_tool_parameter_schema(tool_name: str) -> Dict[str, Any]:
    """
    Get the parameter schema for a specific tool.

    Args:
        tool_name: Name of the tool

    Returns:
        Parameter schema for the tool
    """
    return TOOL_PARAMETER_SCHEMAS.get(tool_name, {})


def validate_tool_parameters(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix tool parameters.

    Args:
        tool_name: Name of the tool
        parameters: Tool parameters to validate

    Returns:
        Dictionary containing validated parameters and any errors
    """
    schema = get_tool_parameter_schema(tool_name)
    if not schema:
        return {"parameters": parameters, "errors": []}

    errors = []
    fixed_parameters = parameters.copy()

    # Special handling for test_financial_data
    if tool_name == "test_financial_data":
        return _validate_test_financial_data_parameters(parameters)
    # Special handling for sec_financial_data
    elif tool_name == "sec_financial_data":
        return _validate_sec_financial_data_parameters(parameters)

    # Generic validation for other tools
    for param_name, param_schema in schema.items():
        if param_schema.get("required", False) and param_name not in parameters:
            errors.append(f"Missing required parameter: {param_name}")

    return {"parameters": fixed_parameters, "errors": errors}


def _validate_test_financial_data_parameters(
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate and fix parameters for the test_financial_data tool.

    Args:
        parameters: Tool parameters to validate

    Returns:
        Dictionary containing validated parameters and any errors
    """
    errors = []
    fixed_parameters = parameters.copy()

    # Check query_type
    query_type = fixed_parameters.get("query_type")
    valid_query_types = ["revenue", "profit", "metrics"]

    if query_type not in valid_query_types:
        errors.append(f"Invalid query_type: {query_type}. Must be one of {valid_query_types}")
        # Default to revenue
        fixed_parameters["query_type"] = "revenue"

    # Ensure parameters is a dictionary
    if "parameters" not in fixed_parameters:
        fixed_parameters["parameters"] = {}
    elif fixed_parameters["parameters"] is None:
        fixed_parameters["parameters"] = {}

    # Check for required ticker parameter
    if "ticker" not in fixed_parameters["parameters"]:
        errors.append("Missing required parameter: ticker")
        # Try to extract ticker from other parameters
        if "ticker" in fixed_parameters:
            fixed_parameters["parameters"]["ticker"] = fixed_parameters["ticker"]
        # Default to AAPL if no ticker is found
        else:
            fixed_parameters["parameters"]["ticker"] = "AAPL"

    return {"parameters": fixed_parameters, "errors": errors}


def _validate_sec_financial_data_parameters(
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate and fix parameters for the sec_financial_data tool.

    Args:
        parameters: Tool parameters to validate

    Returns:
        Dictionary containing validated parameters and any errors
    """
    errors = []
    fixed_parameters = parameters.copy()

    # Check query_type
    query_type = fixed_parameters.get("query_type")
    valid_query_types = [
        "financial_facts",
        "company_info",
        "metrics",
        "time_series",
        "financial_ratios",
        "custom_sql",
    ]

    # Fix common query_type errors
    if query_type == "revenue":
        fixed_parameters["query_type"] = "financial_facts"
        if "parameters" not in fixed_parameters:
            fixed_parameters["parameters"] = {}
        if "metrics" not in fixed_parameters["parameters"]:
            fixed_parameters["parameters"]["metrics"] = ["Revenue"]
    elif query_type == "financial_metrics":
        fixed_parameters["query_type"] = "metrics"
    elif query_type not in valid_query_types:
        errors.append(f"Invalid query_type: {query_type}. Must be one of {valid_query_types}")
        fixed_parameters["query_type"] = "financial_facts"  # Default to financial_facts

    # Ensure parameters is a dictionary
    if "parameters" not in fixed_parameters:
        fixed_parameters["parameters"] = {}
    elif fixed_parameters["parameters"] is None:
        fixed_parameters["parameters"] = {}

    # Check required parameters based on query_type
    query_type = fixed_parameters["query_type"]
    if query_type == "financial_facts":
        if "ticker" not in fixed_parameters["parameters"]:
            errors.append("Missing required parameter for financial_facts: ticker")
            # Try to extract ticker from other parameters
            if "ticker" in fixed_parameters:
                fixed_parameters["parameters"]["ticker"] = fixed_parameters["ticker"]
    elif query_type == "time_series":
        if "ticker" not in fixed_parameters["parameters"]:
            errors.append("Missing required parameter for time_series: ticker")
            if "ticker" in fixed_parameters:
                fixed_parameters["parameters"]["ticker"] = fixed_parameters["ticker"]
        if "metric" not in fixed_parameters["parameters"]:
            errors.append("Missing required parameter for time_series: metric")
    elif query_type == "financial_ratios":
        if "ticker" not in fixed_parameters["parameters"]:
            errors.append("Missing required parameter for financial_ratios: ticker")
            if "ticker" in fixed_parameters:
                fixed_parameters["parameters"]["ticker"] = fixed_parameters["ticker"]
    elif query_type == "custom_sql":
        if "sql_query" not in fixed_parameters["parameters"]:
            errors.append("Missing required parameter for custom_sql: sql_query")

    return {"parameters": fixed_parameters, "errors": errors}


def generate_tool_parameter_prompt(tool_name: str) -> str:
    """
    Generate a prompt for the LLM to help it generate correct tool parameters.

    Args:
        tool_name: Name of the tool

    Returns:
        Prompt string with tool parameter information
    """
    schema = get_tool_parameter_schema(tool_name)
    if not schema:
        return f"No parameter schema available for tool: {tool_name}"

    prompt = f"Tool: {tool_name}\n\nParameter Schema:\n"

    # Special handling for test_financial_data
    if tool_name == "test_financial_data":
        prompt += "query_type (required): Type of financial data query to execute\n"
        prompt += "  Valid values: revenue, profit, metrics\n\n"
        prompt += "parameters (object): Parameters for the query\n"
        prompt += "  - ticker (required): Company ticker symbol\n"
        prompt += "  - start_date (optional): Start date for data retrieval (YYYY-MM-DD)\n"
        prompt += "  - end_date (optional): End date for data retrieval (YYYY-MM-DD)\n\n"

        prompt += "Example:\n"
        prompt += "{\n"
        prompt += '  "query_type": "revenue",\n'
        prompt += '  "parameters": {\n'
        prompt += '    "ticker": "AAPL",\n'
        prompt += '    "start_date": "2022-01-01",\n'
        prompt += '    "end_date": "2023-12-31"\n'
        prompt += "  }\n"
        prompt += "}\n"
    # Special handling for sec_financial_data
    elif tool_name == "sec_financial_data":
        prompt += "query_type (required): Type of financial data query to execute\n"
        prompt += (
            "  Valid values: financial_facts, company_info, metrics, time_series, financial_ratios, custom_sql\n\n"
        )
        prompt += "parameters (object): Parameters for the query\n"

        # Add details for each query type
        prompt += "For query_type = 'financial_facts':\n"
        prompt += "  - ticker (required): Company ticker symbol\n"
        prompt += "  - metrics (optional): List of metrics to retrieve\n"
        prompt += "  - start_date (optional): Start date for data retrieval (YYYY-MM-DD)\n"
        prompt += "  - end_date (optional): End date for data retrieval (YYYY-MM-DD)\n"
        prompt += "  - filing_type (optional): Type of SEC filing (10-K, 10-Q, 8-K)\n\n"

        prompt += "For query_type = 'time_series':\n"
        prompt += "  - ticker (required): Company ticker symbol\n"
        prompt += "  - metric (required): Specific metric for time series data\n"
        prompt += "  - start_date (optional): Start date for data retrieval (YYYY-MM-DD)\n"
        prompt += "  - end_date (optional): End date for data retrieval (YYYY-MM-DD)\n"
        prompt += "  - period (optional): Time period for data aggregation (annual, quarterly)\n\n"

        prompt += "Example:\n"
        prompt += "{\n"
        prompt += '  "query_type": "financial_facts",\n'
        prompt += '  "parameters": {\n'
        prompt += '    "ticker": "AAPL",\n'
        prompt += '    "metrics": ["Revenue", "NetIncome"],\n'
        prompt += '    "start_date": "2022-01-01",\n'
        prompt += '    "end_date": "2023-12-31"\n'
        prompt += "  }\n"
        prompt += "}\n"
    else:
        # Generic parameter prompt for other tools
        for param_name, param_schema in schema.items():
            required = "required" if param_schema.get("required", False) else "optional"
            prompt += f"{param_name} ({required}): {param_schema.get('description', '')}\n"

            if "enum" in param_schema:
                prompt += f"  Valid values: {', '.join(param_schema['enum'])}\n"

            if param_schema.get("type") == "array":
                prompt += f"  Type: array of {param_schema.get('items', {}).get('type', 'string')}\n"

            prompt += "\n"

    return prompt
