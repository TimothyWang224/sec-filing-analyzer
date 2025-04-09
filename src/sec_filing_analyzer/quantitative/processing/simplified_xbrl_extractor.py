"""
Simplified XBRL Extractor Module

This module provides a simplified interface to extract financial data from XBRL filings
using the edgartools library (aliased as edgar).
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

# For data processing
import pandas as pd

# Import standardized edgar utilities
from ..utils import edgar_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedXBRLExtractor:
    """
    A simplified extractor for XBRL data from SEC filings using the edgar package.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the XBRL extractor.

        Args:
            cache_dir: Optional directory to cache XBRL data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/xbrl_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized simplified XBRL extractor with cache at {self.cache_dir}")

    def extract_financials(
        self,
        ticker: str,
        filing_id: str,
        accession_number: str,
        filing_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract financial data from XBRL filings.

        Args:
            ticker: Company ticker symbol
            filing_id: Internal filing ID
            accession_number: SEC accession number
            filing_url: Optional direct URL to the filing

        Returns:
            Dictionary of financial data
        """
        try:
            # Check cache first
            cache_file = self.cache_dir / f"{ticker}_{accession_number.replace('-', '_')}.json"
            if cache_file.exists():
                logger.info(f"Loading XBRL data from cache for {ticker} {accession_number}")
                with open(cache_file, 'r') as f:
                    return json.load(f)

            # Get filing by accession number using standardized utility
            filing_obj = None
            if not filing_url:
                try:
                    # Use standardized utility to get filing by accession number
                    filing_obj = edgar_utils.get_filing_by_accession(ticker, accession_number)

                    if filing_obj:
                        # Use the filing URL from the filing object
                        filing_url = filing_obj.filing_url
                    else:
                        raise ValueError(f"Filing with accession number {accession_number} not found")
                except Exception as e:
                    logger.error(f"Error getting filing by accession number: {e}")
                    raise ValueError(f"Failed to get filing by accession number: {str(e)}")

            logger.info(f"Extracting XBRL data for {ticker} {accession_number} from {filing_url}")

            # If we don't have a filing object yet, get it now
            if not filing_obj:
                try:
                    # Use standardized utility to get filing by accession number
                    filing_obj = edgar_utils.get_filing_by_accession(ticker, accession_number)

                    if not filing_obj:
                        raise ValueError(f"Filing with accession number {accession_number} not found")
                except Exception as e:
                    logger.error(f"Error getting filing by accession number: {e}")
                    raise ValueError(f"Failed to get filing by accession number: {str(e)}")

            # Check if the filing has XBRL data
            if not filing_obj.is_xbrl:
                raise ValueError(f"Filing {accession_number} does not have XBRL data")

            # Get the XBRL data
            try:
                xbrl_data = filing_obj.xbrl

                if not xbrl_data:
                    raise ValueError(f"Failed to extract XBRL data from filing {accession_number}")
            except Exception as e:
                logger.error(f"Error loading XBRL data: {e}")
                return {
                    "filing_id": filing_id,
                    "ticker": ticker,
                    "accession_number": accession_number,
                    "error": f"Failed to load XBRL data: {str(e)}"
                }

            # Extract filing metadata
            filing_date = str(filing_obj.filing_date) if filing_obj.filing_date else None
            filing_type = filing_obj.form
            report_date = str(filing_obj.report_date) if filing_obj.report_date else None

            # Determine fiscal year and quarter
            fiscal_year = int(report_date.split('-')[0]) if report_date and '-' in report_date else None
            fiscal_quarter = self._determine_fiscal_quarter(report_date, filing_type)

            # Initialize financials dictionary
            financials = {
                "filing_id": filing_id,
                "ticker": ticker,
                "accession_number": accession_number,
                "filing_url": filing_url,
                "filing_date": filing_date,
                "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter,
                "filing_type": filing_type,
                "facts": [],
                "metrics": {},
                "statements": {}
            }

            # Extract financial statements
            try:
                # Get the statements from the filing
                if hasattr(filing_obj, 'statements') and filing_obj.statements:
                    statements = filing_obj.statements

                    # Check if statements is a dictionary-like object
                    if hasattr(statements, 'items'):
                        # Process each statement
                        for statement_name, statement_data in statements.items():
                            try:
                                # Convert statement to pandas DataFrame if possible
                                if hasattr(statement_data, 'to_pandas'):
                                    df = statement_data.to_pandas()
                                elif isinstance(statement_data, dict):
                                    df = pd.DataFrame(statement_data)
                                else:
                                    # Skip if we can't convert to DataFrame
                                    continue

                                # Skip empty statements
                                if df.empty:
                                    continue

                                # Add statement to financials
                                financials["statements"][statement_name] = df.to_dict(orient='records')

                                # Process statement data
                                self._process_statement_data(df, statement_name, financials)

                            except Exception as e:
                                logger.warning(f"Error processing statement {statement_name}: {e}")
                else:
                    # Try to extract data from the filing text
                    self._extract_data_from_text(filing_obj, financials)
            except Exception as e:
                logger.warning(f"Error extracting financial statements: {e}")

            # Calculate key financial ratios
            self._calculate_ratios(financials)

            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump(financials, f, indent=2)

            logger.info(f"Extracted {len(financials['facts'])} facts for {ticker} {accession_number}")
            return financials

        except Exception as e:
            logger.error(f"Error extracting XBRL data for {ticker} {accession_number}: {str(e)}")
            return {
                "filing_id": filing_id,
                "ticker": ticker,
                "accession_number": accession_number,
                "error": str(e)
            }



    def _get_filing_type_from_url(self, filing_url: str) -> Optional[str]:
        """Extract filing type from filing URL.

        Args:
            filing_url: URL of the filing

        Returns:
            Filing type (10-K, 10-Q, etc.) or None if not found
        """
        try:
            # Try to extract from URL
            if '10-K' in filing_url:
                return '10-K'
            elif '10-Q' in filing_url:
                return '10-Q'
            elif '8-K' in filing_url:
                return '8-K'
            elif '20-F' in filing_url:
                return '20-F'
            elif '40-F' in filing_url:
                return '40-F'
            elif '6-K' in filing_url:
                return '6-K'

            return None
        except Exception as e:
            logger.warning(f"Error extracting filing type: {e}")
            return None

    def _determine_fiscal_quarter(self, filing_date: Optional[str], filing_type: Optional[str]) -> Optional[int]:
        """Determine fiscal quarter from filing date and type.

        Args:
            filing_date: Filing date in YYYY-MM-DD format
            filing_type: Filing type (10-K, 10-Q, etc.)

        Returns:
            Fiscal quarter (1-4) or None if not determined
        """
        try:
            if not filing_date or not filing_type:
                return None

            # For 10-K, assume it's Q4
            if filing_type == '10-K':
                return 4

            # For 10-Q, determine from month
            if filing_type == '10-Q' and len(filing_date) >= 7:
                month = int(filing_date[5:7])
                # Approximate quarter from month
                return (month - 1) // 3 + 1

            return None
        except Exception as e:
            logger.warning(f"Error determining fiscal quarter: {e}")
            return None

    def _process_statement_data(self, df: pd.DataFrame, statement_name: str, financials: Dict[str, Any]) -> None:
        """Process statement data.

        Args:
            df: DataFrame containing statement data
            statement_name: Name of the statement
            financials: Dictionary to store financial data
        """
        try:
            # Determine statement category
            category = self._get_statement_category(statement_name)

            # Process each row in the statement
            for _, row in df.iterrows():
                try:
                    # Skip rows without a concept
                    if 'concept' not in row or not row['concept']:
                        continue

                    # Get concept name
                    concept = row['concept']

                    # Get value from the first non-concept column
                    value = None
                    for col in df.columns:
                        if col != 'concept' and not pd.isna(row[col]):
                            try:
                                value = float(row[col])
                                break
                            except (ValueError, TypeError):
                                pass

                    # Skip if no value found
                    if value is None:
                        continue

                    # Normalize concept name
                    standard_name = self._normalize_concept_name(concept)

                    # Create fact entry
                    fact = {
                        "xbrl_tag": concept,
                        "metric_name": standard_name,
                        "value": value,
                        "statement": statement_name,
                        "category": category
                    }

                    # Add to facts list
                    financials["facts"].append(fact)

                    # Add to metrics dictionary
                    financials["metrics"][standard_name] = value

                except Exception as e:
                    logger.debug(f"Error processing row: {e}")
        except Exception as e:
            logger.warning(f"Error processing statement: {e}")

    def _normalize_concept_name(self, concept: str) -> str:
        """Normalize concept name.

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

    def _get_statement_category(self, statement_name: str) -> str:
        """Get the category of a statement.

        Args:
            statement_name: Name of the statement

        Returns:
            Category of the statement (income_statement, balance_sheet, cash_flow, equity, other)
        """
        statement_name_lower = statement_name.lower()

        if any(term in statement_name_lower for term in ['income', 'operations', 'earnings']):
            return 'income_statement'
        elif any(term in statement_name_lower for term in ['balance', 'financial position']):
            return 'balance_sheet'
        elif any(term in statement_name_lower for term in ['cash flow', 'cash flows']):
            return 'cash_flow'
        elif any(term in statement_name_lower for term in ['equity', 'stockholders', 'shareholders']):
            return 'equity'
        else:
            return 'other'

    def _extract_data_from_text(self, filing_obj: Any, financials: Dict[str, Any]) -> None:
        """Extract financial data from filing text.

        Args:
            filing_obj: Filing object
            financials: Dictionary to store financial data
        """
        try:
            # Get the filing text
            text = filing_obj.text

            if not text:
                return

            # Extract common financial metrics using regex patterns
            patterns = {
                "revenue": r"(?:Total|Net)\s+[Rr]evenue[s]?\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$",
                "net_income": r"(?:Net|Total)\s+[Ii]ncome\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$",
                "total_assets": r"(?:Total|All)\s+[Aa]ssets\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$",
                "total_liabilities": r"(?:Total|All)\s+[Ll]iabilities\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$",
                "stockholders_equity": r"(?:Total|All)\s+(?:[Ss]tockholders'?|[Ss]hareholders'?)\s+[Ee]quity\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$"
            }

            # Extract metrics
            for metric_name, pattern in patterns.items():
                import re
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
        except Exception as e:
            logger.warning(f"Error extracting data from text: {e}")

    def _calculate_ratios(self, financials: Dict[str, Any]) -> None:
        """Calculate financial ratios from extracted data.

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
