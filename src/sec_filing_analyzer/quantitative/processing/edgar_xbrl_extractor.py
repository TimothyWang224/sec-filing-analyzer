"""
Enhanced XBRL Extractor using the edgar library's built-in capabilities.

This module provides a more robust XBRL extraction implementation that leverages
the edgar library's comprehensive XBRL handling capabilities.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

import pandas as pd

# Import edgar utilities
from ..utils import edgar_utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgarXBRLExtractor:
    """
    Enhanced XBRL extractor that leverages the edgar library's built-in capabilities.

    This class provides methods to extract financial data from SEC filings using
    the edgar library's comprehensive XBRL handling capabilities.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the XBRL extractor.

        Args:
            cache_dir: Optional directory to cache extracted data
        """
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def extract_financials(self, ticker: str, filing_id: str, accession_number: str, filing_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract financial data from an SEC filing using the edgar library.

        Args:
            ticker: Company ticker symbol
            filing_id: Unique identifier for the filing
            accession_number: SEC accession number
            filing_url: Optional URL to the filing

        Returns:
            Dictionary containing extracted financial data
        """
        # Check cache first if cache_dir is provided
        if self.cache_dir:
            cache_file = Path(self.cache_dir) / f"{ticker}_{accession_number}.json"
            if cache_file.exists():
                logger.info(f"Using cached data for {ticker} {accession_number}")
                with open(cache_file, 'r') as f:
                    return json.load(f)

        try:
            # Get the filing using edgar's built-in capabilities
            filing = edgar_utils.get_filing_by_accession(ticker, accession_number)

            if not filing:
                logger.error(f"Filing not found: {ticker} {accession_number}")
                return {
                    "filing_id": filing_id,
                    "ticker": ticker,
                    "accession_number": accession_number,
                    "error": f"Filing not found: {accession_number}"
                }

            # Get filing metadata
            metadata = edgar_utils.get_filing_metadata(filing, ticker)

            # Initialize financials dictionary with metadata
            financials = {
                "filing_id": filing_id,
                "ticker": ticker,
                "accession_number": accession_number,
                "filing_url": metadata.get("filing_url", filing_url),
                "filing_date": metadata.get("filing_date"),
                "filing_type": metadata.get("form")
            }

            # Get XBRL data if available
            try:
                # Check if the filing has XBRL data
                has_xbrl = hasattr(filing, 'is_xbrl') and filing.is_xbrl

                if not has_xbrl:
                    logger.warning(f"Filing {accession_number} does not have XBRL data")
                    return self._extract_data_from_text(filing, financials)

                # Get XBRL data - note that xbrl is a method, not a property
                xbrl_data = filing.xbrl()

                if not xbrl_data:
                    logger.warning(f"No XBRL data found for {ticker} {accession_number}")
                    return self._extract_data_from_text(filing, financials)

                # Extract basic metadata from XBRL
                # Check if these methods exist before calling them
                if hasattr(xbrl_data, 'get_entity_name'):
                    financials["entity_name"] = xbrl_data.get_entity_name()
                if hasattr(xbrl_data, 'get_document_type'):
                    financials["document_type"] = xbrl_data.get_document_type()
                if hasattr(xbrl_data, 'get_document_period'):
                    financials["document_period"] = xbrl_data.get_document_period()
                if hasattr(xbrl_data, 'get_fiscal_year_focus'):
                    financials["fiscal_year"] = xbrl_data.get_fiscal_year_focus()
                if hasattr(xbrl_data, 'get_fiscal_period_focus'):
                    financials["fiscal_period"] = xbrl_data.get_fiscal_period_focus()

                # Extract facts
                facts = []
                metrics = {}
                statements = {}

                # Process facts from the instance
                if hasattr(xbrl_data, 'instance') and hasattr(xbrl_data.instance, 'facts'):
                    # Get all facts
                    all_facts = xbrl_data.instance.facts

                    # Filter for US-GAAP facts
                    for index, row in all_facts.iterrows():
                        # Check if this is a US-GAAP fact
                        concept = index[0] if isinstance(index, tuple) else index
                        if not concept.startswith('us-gaap:'):
                            continue

                        # Get the value for each period
                        for period, period_values in row.items():
                            # Skip if no value
                            if not period_values or not isinstance(period_values, dict):
                                continue

                            # Get the base value (non-dimensional)
                            base_value = period_values.get((), {})
                            if not base_value:
                                continue

                            value = base_value.get('value')
                            if not value:
                                continue

                            # Try to convert value to float
                            try:
                                numeric_value = float(value)
                            except (ValueError, TypeError):
                                numeric_value = None

                            # Create fact entry
                            fact_entry = {
                                "xbrl_tag": concept,
                                "metric_name": self._normalize_concept_name(concept),
                                "value": numeric_value if numeric_value is not None else value,
                                "units": base_value.get('units'),
                                "decimals": base_value.get('decimals'),
                                "period": period,
                                "duration": base_value.get('duration')
                            }

                            # Parse period information
                            if ' to ' in period:
                                start_date, end_date = period.split(' to ')
                                fact_entry["start_date"] = start_date
                                fact_entry["end_date"] = end_date
                                fact_entry["period_type"] = "duration"
                            else:
                                fact_entry["instant"] = period
                                fact_entry["period_type"] = "instant"

                            # Add dimensions if present
                            dimensions = {}
                            for dim_key in period_values.keys():
                                if dim_key != () and isinstance(dim_key, tuple):
                                    # Handle different tuple formats
                                    if len(dim_key) == 2 and isinstance(dim_key[0], str) and isinstance(dim_key[1], str):
                                        # Simple (dimension, member) tuple
                                        dimensions[dim_key[0]] = dim_key[1]
                                    else:
                                        # Complex tuple of (dimension, member) pairs
                                        for item in dim_key:
                                            if isinstance(item, tuple) and len(item) == 2:
                                                dimensions[item[0]] = item[1]

                            if dimensions:
                                fact_entry["dimensions"] = dimensions

                            facts.append(fact_entry)

                            # Add to metrics dictionary (using the last value if multiple periods)
                            metrics[fact_entry["metric_name"]] = fact_entry["value"]

                # Extract financial statements if available
                if hasattr(xbrl_data, 'statements_dict'):
                    # Get all available statements
                    for statement_key, statement_def in xbrl_data.statements_dict.items():
                        # Skip if not a financial statement
                        if not statement_key.startswith('Statement'):
                            continue

                        # Get the statement name
                        statement_name = statement_key.replace('Statement', '').lower()

                        # Map to standard statement names
                        if 'income' in statement_name or 'operations' in statement_name:
                            statement_type = 'income_statement'
                        elif 'balance' in statement_name or 'financial position' in statement_name:
                            statement_type = 'balance_sheet'
                        elif 'cash' in statement_name:
                            statement_type = 'cash_flow'
                        elif 'equity' in statement_name or 'stockholder' in statement_name:
                            statement_type = 'equity'
                        elif 'comprehensive' in statement_name:
                            statement_type = 'comprehensive_income'
                        else:
                            statement_type = statement_name

                        # Get the statement
                        try:
                            statement = xbrl_data.get_statement(statement_key)
                            if statement:
                                statements[statement_type] = self._statement_to_dict(statement)
                        except Exception as e:
                            logger.warning(f"Error getting statement {statement_key}: {e}")

                # Try direct statement methods as fallback
                if 'balance_sheet' not in statements and hasattr(xbrl_data, 'get_balance_sheet'):
                    try:
                        balance_sheet = xbrl_data.get_balance_sheet()
                        if balance_sheet:
                            statements["balance_sheet"] = self._statement_to_dict(balance_sheet)
                    except Exception as e:
                        logger.warning(f"Error getting balance sheet: {e}")

                if 'income_statement' not in statements and hasattr(xbrl_data, 'get_income_statement'):
                    try:
                        income_statement = xbrl_data.get_income_statement()
                        if income_statement:
                            statements["income_statement"] = self._statement_to_dict(income_statement)
                    except Exception as e:
                        logger.warning(f"Error getting income statement: {e}")

                if 'cash_flow' not in statements and hasattr(xbrl_data, 'get_cash_flow_statement'):
                    try:
                        cash_flow = xbrl_data.get_cash_flow_statement()
                        if cash_flow:
                            statements["cash_flow"] = self._statement_to_dict(cash_flow)
                    except Exception as e:
                        logger.warning(f"Error getting cash flow statement: {e}")

                if 'equity' not in statements and hasattr(xbrl_data, 'get_statement_of_changes_in_equity'):
                    try:
                        equity = xbrl_data.get_statement_of_changes_in_equity()
                        if equity:
                            statements["equity"] = self._statement_to_dict(equity)
                    except Exception as e:
                        logger.warning(f"Error getting equity statement: {e}")

                # Add extracted data to financials
                financials["facts"] = facts
                financials["metrics"] = metrics
                financials["statements"] = statements

                # Calculate key financial ratios
                self._calculate_ratios(financials)

            except Exception as e:
                logger.error(f"Error extracting XBRL data: {e}")
                # Fall back to text extraction if XBRL parsing fails
                return self._extract_data_from_text(filing, financials)

            # Cache the results if cache_dir is provided
            if self.cache_dir:
                cache_file = Path(self.cache_dir) / f"{ticker}_{accession_number}.json"
                with open(cache_file, 'w') as f:
                    json.dump(financials, f, indent=2)

            return financials

        except Exception as e:
            logger.error(f"Error extracting financials for {ticker} {accession_number}: {e}")
            return {
                "filing_id": filing_id,
                "ticker": ticker,
                "accession_number": accession_number,
                "error": str(e)
            }

    def _statement_to_dict(self, statement) -> Dict[str, Any]:
        """
        Convert a statement object to a dictionary.

        Args:
            statement: Statement object from edgar library

        Returns:
            Dictionary representation of the statement
        """
        if not statement:
            return {}

        result = {}

        # Extract basic statement metadata
        for attr in ['name', 'label', 'entity', 'display_name', 'role']:
            if hasattr(statement, attr):
                value = getattr(statement, attr)
                if value is not None:
                    result[attr] = value

        # Extract periods
        if hasattr(statement, 'periods'):
            result["periods"] = statement.periods
        elif hasattr(statement, 'durations'):
            result["periods"] = list(statement.durations)

        # Extract line items
        if hasattr(statement, 'line_items'):
            line_items = []
            for item in statement.line_items:
                line_item = {}
                for attr in ['concept', 'label', 'level', 'is_abstract', 'section_type', 'parent_section']:
                    if hasattr(item, attr):
                        value = getattr(item, attr)
                        if value is not None:
                            line_item[attr] = value

                # Extract values
                if hasattr(item, 'values'):
                    values = {}
                    for period, period_values in item.values.items():
                        period_dict = {}
                        for dim_key, dim_value in period_values.items():
                            # Handle empty tuple (base value)
                            if dim_key == ():
                                # Base value
                                period_dict['value'] = dim_value.get('value')
                                period_dict['units'] = dim_value.get('units')
                                period_dict['decimals'] = dim_value.get('decimals')
                            # Handle tuple of dimension-member pairs
                            elif isinstance(dim_key, tuple):
                                # Handle different tuple formats
                                if len(dim_key) == 2 and isinstance(dim_key[0], str) and isinstance(dim_key[1], str):
                                    # Simple (dimension, member) tuple
                                    dim_name = f"{dim_key[0]}:{dim_key[1]}"
                                    period_dict[dim_name] = dim_value.get('value')
                                else:
                                    # Complex tuple of (dimension, member) pairs
                                    dim_parts = []
                                    for item in dim_key:
                                        if isinstance(item, tuple) and len(item) == 2:
                                            dim_parts.append(f"{item[0]}:{item[1]}")
                                    if dim_parts:
                                        dim_name = '_'.join(dim_parts)
                                        period_dict[dim_name] = dim_value.get('value')
                        values[period] = period_dict
                    line_item['values'] = values

                line_items.append(line_item)
            result["line_items"] = line_items

        # Convert DataFrame to dict if available
        if hasattr(statement, 'data') and isinstance(statement.data, pd.DataFrame):
            try:
                # Try to convert to records
                data_records = statement.data.reset_index().to_dict(orient='records')
                result["data"] = data_records
            except Exception as e:
                # If conversion fails, just store the column and index information
                result["data_columns"] = list(statement.data.columns)
                result["data_index"] = list(statement.data.index)

        return result

    def _extract_data_from_text(self, filing, financials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract financial data from filing text when XBRL is not available.

        Args:
            filing: Filing object
            financials: Dictionary to store financial data

        Returns:
            Updated financials dictionary
        """
        try:
            # Get the filing text
            text = filing.text()

            if not text:
                return financials

            # Extract common financial metrics using regex patterns
            import re
            patterns = {
                "revenue": r"(?:Total|Net)\s+[Rr]evenue[s]?\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$",
                "net_income": r"(?:Net|Total)\s+[Ii]ncome\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$",
                "total_assets": r"(?:Total|All)\s+[Aa]ssets\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$",
                "total_liabilities": r"(?:Total|All)\s+[Ll]iabilities\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$",
                "stockholders_equity": r"(?:Total|All)\s+(?:[Ss]tockholders'?|[Ss]hareholders'?)\s+[Ee]quity\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$"
            }

            # Initialize metrics dictionary if not present
            if "metrics" not in financials:
                financials["metrics"] = {}

            # Initialize facts list if not present
            if "facts" not in financials:
                financials["facts"] = []

            # Extract metrics
            for metric_name, pattern in patterns.items():
                matches = re.findall(pattern, text, re.MULTILINE)
                if matches:
                    # Use the first match
                    value_str = matches[0].replace(',', '')
                    try:
                        value = float(value_str)

                        # Add to metrics
                        financials["metrics"][metric_name] = value

                        # Add to facts
                        fact = {
                            "xbrl_tag": metric_name,
                            "metric_name": metric_name,
                            "value": value,
                            "category": "extracted_from_text"
                        }
                        financials["facts"].append(fact)
                    except ValueError:
                        pass

            # Calculate ratios based on extracted metrics
            self._calculate_ratios(financials)

            return financials

        except Exception as e:
            logger.error(f"Error extracting data from text: {e}")
            return financials

    def _normalize_concept_name(self, concept: str) -> str:
        """
        Normalize concept name by removing namespace prefix and converting to snake_case.

        Args:
            concept: XBRL concept name

        Returns:
            Normalized concept name
        """
        # Remove namespace prefix if present
        if ':' in concept:
            concept = concept.split(':')[1]

        # Convert camel case to snake case
        result = ''
        for i, char in enumerate(concept):
            if i > 0 and char.isupper() and concept[i-1].islower():
                result += '_'
            result += char.lower()

        return result

    def _calculate_ratios(self, financials: Dict[str, Any]) -> None:
        """
        Calculate financial ratios from extracted data.

        Args:
            financials: Dictionary of financial data
        """
        metrics = financials.get("metrics", {})
        ratios = {}

        # Gross margin
        if "revenue" in metrics and "cost_of_revenue" in metrics:
            revenue = float(metrics["revenue"])
            cogs = float(metrics["cost_of_revenue"])
            if revenue > 0:
                ratios["gross_margin"] = (revenue - cogs) / revenue

        # Operating margin
        if "revenue" in metrics and "operating_income" in metrics:
            revenue = float(metrics["revenue"])
            operating_income = float(metrics["operating_income"])
            if revenue > 0:
                ratios["operating_margin"] = operating_income / revenue

        # Net margin
        if "revenue" in metrics and "net_income" in metrics:
            revenue = float(metrics["revenue"])
            net_income = float(metrics["net_income"])
            if revenue > 0:
                ratios["net_margin"] = net_income / revenue

        # Current ratio
        if "current_assets" in metrics and "current_liabilities" in metrics:
            current_assets = float(metrics["current_assets"])
            current_liabilities = float(metrics["current_liabilities"])
            if current_liabilities > 0:
                ratios["current_ratio"] = current_assets / current_liabilities

        # Debt to equity
        if "total_liabilities" in metrics and "stockholders_equity" in metrics:
            total_liabilities = float(metrics["total_liabilities"])
            equity = float(metrics["stockholders_equity"])
            if equity > 0:
                ratios["debt_to_equity"] = total_liabilities / equity

        # Return on equity
        if "net_income" in metrics and "stockholders_equity" in metrics:
            net_income = float(metrics["net_income"])
            equity = float(metrics["stockholders_equity"])
            if equity > 0:
                ratios["return_on_equity"] = net_income / equity

        # Return on assets
        if "net_income" in metrics and "total_assets" in metrics:
            net_income = float(metrics["net_income"])
            assets = float(metrics["total_assets"])
            if assets > 0:
                ratios["return_on_assets"] = net_income / assets

        # Add ratios to financials
        financials["ratios"] = ratios
