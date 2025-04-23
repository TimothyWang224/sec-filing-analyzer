"""
LLM-based parameter completion for tools.

This module provides a class for completing tool parameters using an LLM,
which can extract relevant information from user input and context.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

from sec_filing_analyzer.llm import BaseLLM
from ..utils.json_utils import safe_parse_json, repair_json
from .tool_parameter_helper import get_tool_parameter_schema, validate_tool_parameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMParameterCompleter:
    """
    Class for completing tool parameters using an LLM.

    This class uses an LLM to extract relevant information from user input and context
    to complete tool parameters.
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize the parameter completer.

        Args:
            llm: LLM instance to use for parameter completion
        """
        self.llm = llm

    async def complete_parameters(
        self,
        tool_name: str,
        partial_parameters: Dict[str, Any],
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete tool parameters using the LLM.

        Args:
            tool_name: Name of the tool
            partial_parameters: Partial parameters to complete
            user_input: User input to extract information from
            context: Optional additional context

        Returns:
            Completed parameters
        """
        # Get tool parameter schema
        schema = get_tool_parameter_schema(tool_name)
        if not schema:
            return partial_parameters

        # First apply standard validation and fixes
        validation_result = validate_tool_parameters(tool_name, partial_parameters)
        fixed_parameters = validation_result["parameters"]

        # Check if we have an error message in the context
        error_message = None
        if context and "last_error" in context:
            error_message = context["last_error"]

        # Back-fill required parameters from context if possible
        fixed_parameters = self._backfill_required_parameters(tool_name, fixed_parameters, context)

        # If there are no errors and no error message, return the fixed parameters
        if not validation_result["errors"] and not error_message:
            return fixed_parameters

        # Otherwise, try to complete the parameters using the LLM
        try:
            # Create prompt for parameter completion
            prompt = self._create_parameter_completion_prompt(
                tool_name, schema, fixed_parameters, user_input, validation_result["errors"], context
            )

            # Generate completed parameters
            system_prompt = """You are an expert at extracting information from text to complete tool parameters.
Your task is to analyze the user input and extract relevant information to complete the tool parameters.
Return only the completed parameters as a JSON object.
"""

            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,  # Low temperature for more deterministic extraction
                json_mode=True  # Force the model to return valid JSON
            )

            # Parse completed parameters from response
            completed_parameters = await self._parse_parameters(response, fixed_parameters)

            # Apply special handling for specific tools
            if tool_name == "sec_financial_data":
                completed_parameters = await self._enhance_sec_financial_data_parameters(
                    completed_parameters, user_input, context
                )
            elif tool_name == "sec_semantic_search":
                completed_parameters = await self._enhance_sec_semantic_search_parameters(
                    completed_parameters, user_input, context
                )
            elif tool_name == "sec_graph_query":
                completed_parameters = await self._enhance_sec_graph_query_parameters(
                    completed_parameters, user_input, context
                )

            # Validate the completed parameters again
            final_validation = validate_tool_parameters(tool_name, completed_parameters)

            return final_validation["parameters"]

        except Exception as e:
            logger.error(f"Error completing parameters: {str(e)}")
            return fixed_parameters  # Return the original fixed parameters if completion fails

    def _create_parameter_completion_prompt(
        self,
        tool_name: str,
        schema: Dict[str, Any],
        partial_parameters: Dict[str, Any],
        user_input: str,
        errors: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a prompt for parameter completion."""
        # Format the schema for the prompt
        schema_str = json.dumps(schema, indent=2)

        # Format the partial parameters for the prompt
        partial_params_str = json.dumps(partial_parameters, indent=2)

        # Format the errors for the prompt
        errors_str = "\n".join([f"- {error}" for error in errors])

        # Check if we have an error message in the context
        error_message = None
        if context and "last_error" in context:
            error_message = context.get("last_error")
        error_section = ""
        if error_message:
            error_section = f"""

Previous Error:
{error_message}

Please fix the parameters to address this error.
"""

        # Check if we had identical errors in previous attempts
        identical_errors_section = ""
        if context and context.get("identical_errors", False):
            identical_errors_section = """

IMPORTANT: Previous attempts to fix this parameter have failed with identical errors.
Pay special attention to the schema requirements and ensure all required fields are present.
"""

        # Include full tool schema if available
        tool_schema_section = ""
        if context and "tool_schema" in context:
            tool_schema_str = json.dumps(context.get("tool_schema", {}), indent=2)
            tool_schema_section = f"""

Complete Tool Schema:
{tool_schema_str}
"""

        return f"""
User Input: {user_input}

Tool: {tool_name}

Parameter Schema:
{schema_str}

Current Parameters:
{partial_params_str}

Parameter Errors:
{errors_str}{error_section}{identical_errors_section}{tool_schema_section}

Please complete the parameters based on the user input and schema.
Extract any relevant information from the user input to fill in missing parameters.
For dates, use YYYY-MM-DD format.
For company names, extract both the name and ticker if possible.
For metrics, identify specific financial metrics mentioned.

Return only the completed parameters as a JSON object.
"""

    async def _parse_parameters(self, response: str, default_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Parse parameters from LLM response."""
        try:
            # First try to parse using the safe_parse_json utility
            completed_parameters = safe_parse_json(response, default_value={}, expected_type="object")

            # If parsing failed and we have an LLM instance, try to repair
            if not completed_parameters and hasattr(self, 'llm'):
                logger.info("Attempting to repair JSON parameters")
                completed_parameters = await repair_json(response, self.llm, default_value={}, expected_type="object")

            # Merge with default parameters
            merged_parameters = default_parameters.copy()
            self._deep_update(merged_parameters, completed_parameters)

            return merged_parameters
        except Exception as e:
            logger.error(f"Error parsing parameters: {str(e)}")
            logger.error(f"Response: {response}")
            return default_parameters  # Return default parameters if parsing fails

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep update a nested dictionary."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def _backfill_required_parameters(self, tool_name: str, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Back-fill required parameters from context."""
        if not context:
            return parameters

        # Create a copy of the parameters to avoid modifying the original
        enhanced_params = parameters.copy()

        # Handle sec_graph_query tool specifically
        if tool_name == "sec_graph_query":
            query_type = enhanced_params.get("query_type")

            # For related_companies query, ensure ticker parameter is present
            if query_type == "related_companies":
                # Initialize parameters dict if it doesn't exist
                if "parameters" not in enhanced_params:
                    enhanced_params["parameters"] = {}
                elif enhanced_params["parameters"] is None:
                    enhanced_params["parameters"] = {}

                # If ticker is missing, try to get it from context
                if "ticker" not in enhanced_params["parameters"]:
                    # Check if we have company info in context
                    if "company_info" in context and "ticker" in context["company_info"]:
                        enhanced_params["parameters"]["ticker"] = context["company_info"]["ticker"]
                    # Check if we have tool results with companies
                    elif "tool_results" in context:
                        for result in context["tool_results"]:
                            if result.get("tool") == "sec_financial_data" and result.get("success", False):
                                # Look for companies query result
                                if result.get("result", {}).get("query_type") == "companies":
                                    companies = result.get("result", {}).get("results", [])
                                    if companies and len(companies) > 0:
                                        # Use the first company ticker
                                        enhanced_params["parameters"]["ticker"] = companies[0].get("ticker")
                                        break

        return enhanced_params

    async def _enhance_sec_graph_query_parameters(
        self,
        parameters: Dict[str, Any],
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhance parameters for the sec_graph_query tool."""
        enhanced_params = parameters.copy()

        # Ensure parameters is a dictionary
        if "parameters" not in enhanced_params:
            enhanced_params["parameters"] = {}
        elif enhanced_params["parameters"] is None:
            enhanced_params["parameters"] = {}

        # Extract company information
        company_info = await self._extract_company_info(user_input, context)
        if company_info.get("ticker") and "ticker" not in enhanced_params["parameters"]:
            enhanced_params["parameters"]["ticker"] = company_info["ticker"]

        # If we have a query_type but no parameters, try to extract relevant information
        query_type = enhanced_params.get("query_type")
        if query_type:
            # For related_companies, ensure we have a ticker
            if query_type == "related_companies" and "ticker" not in enhanced_params["parameters"]:
                # If we still don't have a ticker, try to get it from previous tool results
                if context and "tool_results" in context:
                    for result in context["tool_results"]:
                        if result.get("tool") == "sec_financial_data" and result.get("success", False):
                            # Look for companies query result
                            if result.get("result", {}).get("query_type") == "companies":
                                companies = result.get("result", {}).get("results", [])
                                if companies and len(companies) > 0:
                                    # Use the first company ticker
                                    enhanced_params["parameters"]["ticker"] = companies[0].get("ticker")
                                    break

        return enhanced_params

    async def _enhance_sec_financial_data_parameters(
        self,
        parameters: Dict[str, Any],
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhance parameters for the sec_financial_data tool."""
        enhanced_params = parameters.copy()

        # Ensure parameters is a dictionary
        if "parameters" not in enhanced_params:
            enhanced_params["parameters"] = {}
        elif enhanced_params["parameters"] is None:
            enhanced_params["parameters"] = {}

        # Extract company information
        company_info = await self._extract_company_info(user_input, context)
        if company_info.get("ticker") and "ticker" not in enhanced_params["parameters"]:
            enhanced_params["parameters"]["ticker"] = company_info["ticker"]

        # Extract date range
        date_range = self._extract_date_range(user_input)
        if date_range.get("start_date") and "start_date" not in enhanced_params["parameters"]:
            enhanced_params["parameters"]["start_date"] = date_range["start_date"]
        if date_range.get("end_date") and "end_date" not in enhanced_params["parameters"]:
            enhanced_params["parameters"]["end_date"] = date_range["end_date"]

        # Extract metrics
        metrics = await self._extract_financial_metrics(user_input, enhanced_params.get("query_type", ""))
        if metrics and "metrics" not in enhanced_params["parameters"]:
            enhanced_params["parameters"]["metrics"] = metrics

        return enhanced_params

    async def _enhance_sec_semantic_search_parameters(
        self,
        parameters: Dict[str, Any],
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhance parameters for the sec_semantic_search tool."""
        enhanced_params = parameters.copy()

        # Ensure query parameter is always provided
        if "query" not in enhanced_params:
            enhanced_params["query"] = user_input
            logger.info(f"Added missing query parameter to sec_semantic_search tool: {user_input}")

        # Extract company information
        company_info = await self._extract_company_info(user_input, context)
        if company_info.get("ticker") and "companies" not in enhanced_params:
            enhanced_params["companies"] = [company_info["ticker"]]

        # Extract date range
        date_range = self._extract_date_range(user_input)
        if date_range.get("start_date") and date_range.get("end_date") and "date_range" not in enhanced_params:
            enhanced_params["date_range"] = [date_range["start_date"], date_range["end_date"]]

        # Extract filing types
        filing_types = self._extract_filing_types(user_input)
        if filing_types and "filing_types" not in enhanced_params:
            enhanced_params["filing_types"] = filing_types

        # If query is missing, use the user input as the query
        if "query" not in enhanced_params or not enhanced_params["query"]:
            enhanced_params["query"] = user_input

        return enhanced_params

    async def _extract_company_info(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract company information from text."""
        # Check if we already have company information in the context
        if context and "company_info" in context:
            return context["company_info"]

        # Use the LLM to extract company information
        prompt = f"""
Extract company information from the following text:

{text}

Return a JSON object with the following fields:
- company_name: The name of the company
- ticker: The ticker symbol of the company

If multiple companies are mentioned, focus on the main company being discussed.
If no company is mentioned, return null for both fields.
"""

        system_prompt = """You are an expert at extracting company information from text.
Your task is to analyze the text and extract the company name and ticker symbol.
Return only the extracted information as a JSON object.
"""

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            json_mode=True  # Force the model to return valid JSON
        )

        try:
            # Parse the JSON using our safe_parse_json utility
            company_info = safe_parse_json(response, default_value={}, expected_type="object")

            # If parsing failed and we have an LLM instance, try to repair
            if not company_info:
                logger.info("Attempting to repair JSON company info")
                company_info = await repair_json(response, self.llm, default_value={}, expected_type="object")

            # If we have a company name but no ticker, try to resolve the ticker
            if company_info.get("company_name") and not company_info.get("ticker"):
                company_info["ticker"] = await self._resolve_ticker(company_info["company_name"])

            return company_info
        except Exception as e:
            logger.error(f"Error extracting company information: {str(e)}")
            return {}

    async def _resolve_ticker(self, company_name: str) -> Optional[str]:
        """Resolve a company name to a ticker symbol."""
        # Common company name to ticker mappings
        common_tickers = {
            "apple": "AAPL",
            "microsoft": "MSFT",
            "amazon": "AMZN",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "facebook": "META",
            "meta": "META",
            "tesla": "TSLA",
            "netflix": "NFLX",
            "nvidia": "NVDA",
            "walmart": "WMT",
            "jpmorgan": "JPM",
            "jp morgan": "JPM",
            "bank of america": "BAC",
            "exxon": "XOM",
            "exxonmobil": "XOM",
            "johnson & johnson": "JNJ",
            "procter & gamble": "PG",
            "coca-cola": "KO",
            "coca cola": "KO",
            "disney": "DIS",
            "verizon": "VZ",
            "at&t": "T",
            "intel": "INTC",
            "ibm": "IBM",
            "cisco": "CSCO",
            "pfizer": "PFE",
            "merck": "MRK",
            "home depot": "HD",
            "mcdonald's": "MCD",
            "mcdonalds": "MCD",
            "goldman sachs": "GS",
            "boeing": "BA",
            "3m": "MMM",
            "caterpillar": "CAT",
            "chevron": "CVX",
            "visa": "V",
            "mastercard": "MA",
            "american express": "AXP",
            "amex": "AXP",
            "nike": "NKE",
            "starbucks": "SBUX",
            "adobe": "ADBE",
            "salesforce": "CRM",
            "oracle": "ORCL",
            "paypal": "PYPL",
            "qualcomm": "QCOM",
            "comcast": "CMCSA",
            "pepsico": "PEP",
            "pepsi": "PEP",
            "ups": "UPS",
            "fedex": "FDX",
            "target": "TGT",
            "costco": "COST",
            "lowes": "LOW",
            "lowe's": "LOW"
        }

        # Check if the company name is in the common tickers
        company_lower = company_name.lower()
        if company_lower in common_tickers:
            return common_tickers[company_lower]

        # Use the LLM to resolve the ticker
        prompt = f"""
What is the ticker symbol for {company_name}?

Return only the ticker symbol as a string.
"""

        system_prompt = """You are an expert at resolving company names to ticker symbols.
Your task is to determine the ticker symbol for the given company.
Return only the ticker symbol as a string.
"""

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            json_mode=False  # Don't use JSON mode for ticker resolution (we want a simple string)
        )

        # Extract ticker from response
        ticker_match = re.search(r'[A-Z]{1,5}', response)
        if ticker_match:
            return ticker_match.group(0)

        return None

    def _extract_date_range(self, text: str) -> Dict[str, str]:
        """Extract date range from text."""
        # Initialize result
        result = {}

        # Try to extract specific dates
        date_pattern = r'(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{2})'
        dates = re.findall(date_pattern, text)

        # Normalize dates to YYYY-MM-DD format
        normalized_dates = []
        for date_str in dates:
            try:
                if '-' in date_str:
                    # Already in YYYY-MM-DD format
                    normalized_dates.append(date_str)
                elif '/' in date_str:
                    # Convert MM/DD/YYYY or MM/DD/YY to YYYY-MM-DD
                    parts = date_str.split('/')
                    if len(parts) == 3:
                        month, day, year = parts
                        if len(year) == 2:
                            year = '20' + year  # Assume 20xx for 2-digit years
                        normalized_dates.append(f"{year}-{month.zfill(2)}-{day.zfill(2)}")
            except Exception:
                pass

        # If we have at least two dates, use the earliest as start_date and latest as end_date
        if len(normalized_dates) >= 2:
            normalized_dates.sort()
            result["start_date"] = normalized_dates[0]
            result["end_date"] = normalized_dates[-1]

        # If we don't have specific dates, try to extract year references
        if not result:
            year_pattern = r'\b(20\d{2})\b'
            years = re.findall(year_pattern, text)

            if years:
                years.sort()

                # If we have at least two years, use the earliest as start_date and latest as end_date
                if len(years) >= 2:
                    result["start_date"] = f"{years[0]}-01-01"
                    result["end_date"] = f"{years[-1]}-12-31"
                # If we have only one year, use it for both start_date and end_date
                else:
                    result["start_date"] = f"{years[0]}-01-01"
                    result["end_date"] = f"{years[0]}-12-31"

        # If we still don't have dates, check for relative time references
        if not result:
            # Check for "last X years"
            last_years_match = re.search(r'last\s+(\d+)\s+years?', text, re.IGNORECASE)
            if last_years_match:
                years = int(last_years_match.group(1))
                end_date = datetime.now()
                start_date = end_date - timedelta(days=years*365)
                result["start_date"] = start_date.strftime("%Y-%m-%d")
                result["end_date"] = end_date.strftime("%Y-%m-%d")

            # Check for "since YYYY"
            since_year_match = re.search(r'since\s+(20\d{2})', text, re.IGNORECASE)
            if since_year_match:
                year = since_year_match.group(1)
                result["start_date"] = f"{year}-01-01"
                result["end_date"] = datetime.now().strftime("%Y-%m-%d")

        # If we still don't have dates, use default values
        if not result:
            # Default to last 3 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3*365)
            result["start_date"] = start_date.strftime("%Y-%m-%d")
            result["end_date"] = end_date.strftime("%Y-%m-%d")

        return result

    def _extract_filing_types(self, text: str) -> List[str]:
        """Extract filing types from text."""
        filing_types = []

        # Check for common filing type references
        if re.search(r'\b10-K\b|\bannual\b|\bannual\s+report\b', text, re.IGNORECASE):
            filing_types.append("10-K")

        if re.search(r'\b10-Q\b|\bquarterly\b|\bquarterly\s+report\b', text, re.IGNORECASE):
            filing_types.append("10-Q")

        if re.search(r'\b8-K\b|\bcurrent\b|\bcurrent\s+report\b', text, re.IGNORECASE):
            filing_types.append("8-K")

        # If no specific filing types are mentioned, default to 10-K and 10-Q
        if not filing_types:
            filing_types = ["10-K", "10-Q"]

        return filing_types

    async def _extract_financial_metrics(self, text: str, query_type: str) -> List[str]:
        """Extract financial metrics from text."""
        # If query_type is not financial_facts, return empty list
        if query_type != "financial_facts":
            return []

        # Common financial metrics
        common_metrics = {
            "revenue": ["Revenue", "Sales", "TotalRevenue"],
            "net income": ["NetIncome", "Profit", "NetEarnings"],
            "earnings": ["EPS", "EarningsPerShare", "DilutedEPS"],
            "assets": ["TotalAssets", "Assets"],
            "liabilities": ["TotalLiabilities", "Liabilities"],
            "equity": ["StockholdersEquity", "Equity", "TotalEquity"],
            "cash": ["Cash", "CashAndCashEquivalents"],
            "debt": ["TotalDebt", "LongTermDebt"],
            "operating income": ["OperatingIncome", "IncomeFromOperations"],
            "gross profit": ["GrossProfit", "GrossMargin"],
            "ebitda": ["EBITDA", "EarningsBeforeInterestTaxesDepreciationAmortization"],
            "free cash flow": ["FreeCashFlow", "FCF"],
            "dividend": ["Dividend", "DividendPerShare"],
            "capex": ["CapitalExpenditures", "CapEx"],
            "r&d": ["ResearchAndDevelopment", "R&D"],
            "inventory": ["Inventory", "TotalInventory"],
            "accounts receivable": ["AccountsReceivable", "AR"],
            "accounts payable": ["AccountsPayable", "AP"]
        }

        # Check for common metric references
        metrics = []
        for keyword, metric_names in common_metrics.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                metrics.append(metric_names[0])  # Add the primary metric name

        # If no specific metrics are mentioned, use the LLM to extract metrics
        if not metrics:
            prompt = f"""
Extract financial metrics from the following text:

{text}

Return a JSON array of financial metric names.
Use standard financial metric names like Revenue, NetIncome, EPS, etc.
If no specific metrics are mentioned, return the most relevant metrics based on the context.
"""

            system_prompt = """You are an expert at extracting financial metrics from text.
Your task is to analyze the text and extract the financial metrics being discussed.
Return only the extracted metrics as a JSON array.
"""

            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                json_mode=True  # Force the model to return valid JSON
            )

            try:
                # Parse the JSON using our safe_parse_json utility
                metrics = safe_parse_json(response, default_value=[], expected_type="array")

                # If parsing failed, try to repair
                if not metrics:
                    logger.info("Attempting to repair JSON metrics")
                    metrics = await repair_json(response, self.llm, default_value=[], expected_type="array")
            except Exception as e:
                logger.error(f"Error extracting financial metrics: {str(e)}")
                # Default to Revenue and NetIncome if extraction fails
                metrics = ["Revenue", "NetIncome"]

        return metrics
