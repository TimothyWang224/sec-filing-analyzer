"""
Simplified XBRL Extractor using the edgar library's built-in capabilities.

This module provides a more focused XBRL extraction implementation that leverages
the edgar library's XBRL handling capabilities.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import edgar

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgarXBRLExtractorSimple:
    """
    Simplified XBRL extractor that leverages the edgar library's built-in capabilities.

    This class provides methods to extract financial data from SEC filings using
    the edgar library's XBRL handling capabilities.
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

    def extract_financials(self, ticker: str, filing_id: str, accession_number: str) -> Dict[str, Any]:
        """
        Extract financial data from an SEC filing using the edgar library.

        Args:
            ticker: Company ticker symbol
            filing_id: Unique identifier for the filing
            accession_number: SEC accession number

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
            # Get the entity
            entity = edgar.get_entity(ticker)
            if not entity:
                logger.error(f"Entity not found: {ticker}")
                return {
                    "filing_id": filing_id,
                    "ticker": ticker,
                    "accession_number": accession_number,
                    "error": f"Entity not found: {ticker}"
                }

            # Get all filings
            filings = entity.get_filings()

            # Find the filing with the matching accession number
            filing = None
            for f in filings:
                if f.accession_number == accession_number:
                    filing = f
                    break

            if not filing:
                logger.error(f"Filing not found: {ticker} {accession_number}")
                return {
                    "filing_id": filing_id,
                    "ticker": ticker,
                    "accession_number": accession_number,
                    "error": f"Filing not found: {accession_number}"
                }

            # Initialize financials dictionary with metadata
            financials = {
                "filing_id": filing_id,
                "ticker": ticker,
                "accession_number": accession_number,
                "filing_url": filing.filing_url,
                "filing_date": str(filing.filing_date),
                "filing_type": filing.form
            }

            # Check if the filing has XBRL data
            has_xbrl = hasattr(filing, 'is_xbrl') and filing.is_xbrl

            if not has_xbrl:
                logger.warning(f"Filing {accession_number} does not have XBRL data")
                return financials

            # Get XBRL data
            try:
                logger.info(f"Getting XBRL data for {ticker} {accession_number}...")
                xbrl_data = filing.xbrl()

                if not xbrl_data:
                    logger.warning(f"No XBRL data found for {ticker} {accession_number}")
                    return financials

                logger.info(f"XBRL data type: {type(xbrl_data)}")
                logger.info(f"XBRL data attributes: {dir(xbrl_data)}")

                # Extract basic metadata from XBRL
                if hasattr(xbrl_data, 'company'):
                    financials["company"] = xbrl_data.company

                # Extract financial statements
                statements = {}
                logger.info("Checking for financial statements...")

                # Check if statements_dict is available
                if hasattr(xbrl_data, 'statements_dict'):
                    logger.info(f"Found statements_dict with {len(xbrl_data.statements_dict)} statements")

                    # Log all statement keys
                    for key in xbrl_data.statements_dict.keys():
                        logger.info(f"Statement key: {key}")

                    # Try to extract statements by type
                    statement_types = {
                        'balance_sheet': ['balance', 'financial position'],
                        'income_statement': ['income', 'operations', 'earnings'],
                        'cash_flow': ['cash flow']
                    }

                    for statement_type, keywords in statement_types.items():
                        try:
                            # Find matching statement keys
                            matching_keys = []
                            for key in xbrl_data.statements_dict.keys():
                                key_lower = key.lower()
                                if any(keyword in key_lower for keyword in keywords):
                                    matching_keys.append(key)

                            if matching_keys:
                                logger.info(f"Found {len(matching_keys)} matching keys for {statement_type}: {matching_keys}")

                                # Use the first matching key
                                statement_key = matching_keys[0]

                                # Get the statement
                                statement = xbrl_data.get_statement(statement_key)
                                if statement:
                                    logger.info(f"Successfully extracted {statement_type}")
                                    statements[statement_type] = self._extract_statement_data(statement)
                        except Exception as e:
                            logger.warning(f"Error extracting {statement_type}: {e}")
                else:
                    logger.warning("No statements_dict found in XBRL data")

                    # Try direct methods as fallback
                    for method_name, statement_type in [
                        ('get_balance_sheet', 'balance_sheet'),
                        ('get_income_statement', 'income_statement'),
                        ('get_cash_flow_statement', 'cash_flow')
                    ]:
                        try:
                            if hasattr(xbrl_data, method_name):
                                logger.info(f"Trying {method_name}...")
                                statement = getattr(xbrl_data, method_name)()
                                if statement:
                                    logger.info(f"Successfully extracted {statement_type} using {method_name}")
                                    statements[statement_type] = self._extract_statement_data(statement)
                        except Exception as e:
                            logger.warning(f"Error extracting {statement_type} using {method_name}: {e}")

                # Add statements to financials
                financials["statements"] = statements

                # Extract key metrics from statements
                metrics = self._extract_key_metrics(statements)
                financials["metrics"] = metrics

                # Cache the results if cache_dir is provided
                if self.cache_dir:
                    cache_file = Path(self.cache_dir) / f"{ticker}_{accession_number}.json"
                    with open(cache_file, 'w') as f:
                        json.dump(financials, f, indent=2, default=str)

                return financials

            except Exception as e:
                logger.error(f"Error extracting XBRL data: {e}")
                return financials

        except Exception as e:
            logger.error(f"Error extracting financials for {ticker} {accession_number}: {e}")
            return {
                "filing_id": filing_id,
                "ticker": ticker,
                "accession_number": accession_number,
                "error": str(e)
            }

    def _extract_statement_data(self, statement) -> Dict[str, Any]:
        """
        Extract data from a statement object.

        Args:
            statement: Statement object from edgar library

        Returns:
            Dictionary representation of the statement
        """
        if not statement:
            return {}

        result = {}

        # Extract basic statement metadata
        for attr in ['name', 'label', 'entity', 'role']:
            if hasattr(statement, attr):
                value = getattr(statement, attr)
                if value is not None:
                    result[attr] = value

        # Extract periods
        if hasattr(statement, 'periods'):
            result["periods"] = statement.periods

        # Extract line items
        if hasattr(statement, 'line_items'):
            line_items = []
            for item in statement.line_items:
                line_item = {}

                # Extract basic line item metadata
                for attr in ['concept', 'label', 'level', 'is_abstract']:
                    if hasattr(item, attr):
                        value = getattr(item, attr)
                        if value is not None:
                            line_item[attr] = value

                # Extract values
                if hasattr(item, 'values') and item.values:
                    values = {}
                    for period, period_values in item.values.items():
                        if () in period_values:
                            # Get base value
                            base_value = period_values[()]
                            if 'value' in base_value:
                                values[period] = base_value['value']

                    if values:
                        line_item['values'] = values

                line_items.append(line_item)

            result["line_items"] = line_items

        return result

    def _extract_key_metrics(self, statements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metrics from financial statements.

        Args:
            statements: Dictionary of financial statements

        Returns:
            Dictionary of key metrics
        """
        metrics = {}

        # Extract metrics from balance sheet
        if 'balance_sheet' in statements:
            balance_sheet = statements['balance_sheet']
            if 'line_items' in balance_sheet:
                for item in balance_sheet['line_items']:
                    concept = item.get('concept', '')

                    # Skip if no concept or values
                    if not concept or 'values' not in item:
                        continue

                    # Get the most recent value
                    values = item['values']
                    if values:
                        # Get the most recent period
                        periods = list(values.keys())
                        periods.sort(reverse=True)
                        if periods:
                            latest_period = periods[0]
                            value = values[latest_period]

                            # Try to convert to float
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                pass

                            # Map common balance sheet concepts
                            if 'Assets' in concept and 'Total' in concept:
                                metrics['total_assets'] = value
                            elif 'Liabilities' in concept and 'Total' in concept:
                                metrics['total_liabilities'] = value
                            elif ('Equity' in concept or 'Stockholders' in concept) and 'Total' in concept:
                                metrics['total_equity'] = value
                            elif 'Cash' in concept and 'Equivalent' in concept:
                                metrics['cash_and_equivalents'] = value
                            elif 'Assets' in concept and 'Current' in concept and 'Total' in concept:
                                metrics['current_assets'] = value
                            elif 'Liabilities' in concept and 'Current' in concept and 'Total' in concept:
                                metrics['current_liabilities'] = value

        # Extract metrics from income statement
        if 'income_statement' in statements:
            income_statement = statements['income_statement']
            if 'line_items' in income_statement:
                for item in income_statement['line_items']:
                    concept = item.get('concept', '')

                    # Skip if no concept or values
                    if not concept or 'values' not in item:
                        continue

                    # Get the most recent value
                    values = item['values']
                    if values:
                        # Get the most recent period
                        periods = list(values.keys())
                        periods.sort(reverse=True)
                        if periods:
                            latest_period = periods[0]
                            value = values[latest_period]

                            # Try to convert to float
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                pass

                            # Map common income statement concepts
                            if 'Revenue' in concept and 'Total' in concept:
                                metrics['revenue'] = value
                            elif 'Income' in concept and 'Net' in concept:
                                metrics['net_income'] = value
                            elif 'Income' in concept and 'Operating' in concept:
                                metrics['operating_income'] = value
                            elif 'Expense' in concept and 'Total' in concept:
                                metrics['total_expenses'] = value
                            elif 'Gross' in concept and 'Profit' in concept:
                                metrics['gross_profit'] = value

        # Extract metrics from cash flow statement
        if 'cash_flow' in statements:
            cash_flow = statements['cash_flow']
            if 'line_items' in cash_flow:
                for item in cash_flow['line_items']:
                    concept = item.get('concept', '')

                    # Skip if no concept or values
                    if not concept or 'values' not in item:
                        continue

                    # Get the most recent value
                    values = item['values']
                    if values:
                        # Get the most recent period
                        periods = list(values.keys())
                        periods.sort(reverse=True)
                        if periods:
                            latest_period = periods[0]
                            value = values[latest_period]

                            # Try to convert to float
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                pass

                            # Map common cash flow concepts
                            if 'Cash' in concept and 'Operating' in concept and 'Net' in concept:
                                metrics['operating_cash_flow'] = value
                            elif 'Cash' in concept and 'Investing' in concept and 'Net' in concept:
                                metrics['investing_cash_flow'] = value
                            elif 'Cash' in concept and 'Financing' in concept and 'Net' in concept:
                                metrics['financing_cash_flow'] = value
                            elif 'Capital' in concept and 'Expenditure' in concept:
                                metrics['capital_expenditures'] = value

        # Calculate financial ratios
        self._calculate_ratios(metrics)

        return metrics

    def _calculate_ratios(self, metrics: Dict[str, Any]) -> None:
        """
        Calculate financial ratios from extracted metrics.

        Args:
            metrics: Dictionary of financial metrics
        """
        # Current ratio
        if 'current_assets' in metrics and 'current_liabilities' in metrics:
            current_assets = float(metrics['current_assets'])
            current_liabilities = float(metrics['current_liabilities'])
            if current_liabilities > 0:
                metrics['current_ratio'] = current_assets / current_liabilities

        # Debt to equity ratio
        if 'total_liabilities' in metrics and 'total_equity' in metrics:
            total_liabilities = float(metrics['total_liabilities'])
            total_equity = float(metrics['total_equity'])
            if total_equity > 0:
                metrics['debt_to_equity'] = total_liabilities / total_equity

        # Return on assets
        if 'net_income' in metrics and 'total_assets' in metrics:
            net_income = float(metrics['net_income'])
            total_assets = float(metrics['total_assets'])
            if total_assets > 0:
                metrics['return_on_assets'] = net_income / total_assets

        # Return on equity
        if 'net_income' in metrics and 'total_equity' in metrics:
            net_income = float(metrics['net_income'])
            total_equity = float(metrics['total_equity'])
            if total_equity > 0:
                metrics['return_on_equity'] = net_income / total_equity

        # Profit margin
        if 'net_income' in metrics and 'revenue' in metrics:
            net_income = float(metrics['net_income'])
            revenue = float(metrics['revenue'])
            if revenue > 0:
                metrics['profit_margin'] = net_income / revenue

        # Operating margin
        if 'operating_income' in metrics and 'revenue' in metrics:
            operating_income = float(metrics['operating_income'])
            revenue = float(metrics['revenue'])
            if revenue > 0:
                metrics['operating_margin'] = operating_income / revenue
