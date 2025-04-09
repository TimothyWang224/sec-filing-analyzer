"""
XBRL Extractor Module

This module provides functionality to extract financial data from XBRL filings
using the edgartools library (aliased as edgar).
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

# For data processing
import pandas as pd

# Import edgartools components (aliased as edgar)
import edgar
from edgar.xbrl import XBRLData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XBRLExtractor:
    """
    Extracts financial data from XBRL filings using edgartools (aliased as edgar).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the XBRL extractor.

        Args:
            cache_dir: Optional directory to cache XBRL data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/xbrl_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized XBRL extractor with cache at {self.cache_dir}")

        # Load standard XBRL tag mappings
        self.tag_mappings = self._load_standard_tag_mappings()

    def _load_standard_tag_mappings(self) -> Dict[str, Dict[str, str]]:
        """Load standard XBRL tag mappings.

        Returns:
            Dictionary mapping XBRL tags to standardized metric names
        """
        # This is a simplified version - in production, load from a file or database
        mappings = {}

        # Income Statement
        income_statement_tags = [
            "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet", "CostOfRevenue", "CostOfGoodsAndServicesSold",
            "GrossProfit", "OperatingExpenses", "ResearchAndDevelopmentExpense",
            "SellingGeneralAndAdministrativeExpense", "OperatingIncomeLoss",
            "NonoperatingIncomeExpense", "InterestExpense", "IncomeTaxExpenseBenefit",
            "NetIncomeLoss", "EarningsPerShareBasic", "EarningsPerShareDiluted"
        ]

        # Balance Sheet
        balance_sheet_tags = [
            "Assets", "AssetsCurrent", "CashAndCashEquivalentsAtCarryingValue",
            "ShortTermInvestments", "AccountsReceivableNetCurrent", "InventoryNet",
            "AssetsNoncurrent", "PropertyPlantAndEquipmentNet", "Goodwill",
            "IntangibleAssetsNetExcludingGoodwill", "LongTermInvestments",
            "Liabilities", "LiabilitiesCurrent", "AccountsPayableCurrent",
            "AccruedLiabilitiesCurrent", "LongTermDebtCurrent", "LiabilitiesNoncurrent",
            "LongTermDebtNoncurrent", "StockholdersEquity", "CommonStock",
            "AdditionalPaidInCapital", "RetainedEarningsAccumulatedDeficit",
            "AccumulatedOtherComprehensiveIncomeLossNetOfTax", "TreasuryStockValue"
        ]

        # Cash Flow
        cash_flow_tags = [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInInvestingActivities",
            "NetCashProvidedByUsedInFinancingActivities",
            "DepreciationDepletionAndAmortization",
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "ProceedsFromIssuanceOfCommonStock",
            "PaymentsOfDividends",
            "PaymentsForRepurchaseOfCommonStock",
            "CashAndCashEquivalentsPeriodIncreaseDecrease"
        ]

        # Create mappings
        for tag in income_statement_tags:
            mappings[tag] = {
                "standard_name": self._normalize_tag(tag),
                "category": "income_statement"
            }

        for tag in balance_sheet_tags:
            mappings[tag] = {
                "standard_name": self._normalize_tag(tag),
                "category": "balance_sheet"
            }

        for tag in cash_flow_tags:
            mappings[tag] = {
                "standard_name": self._normalize_tag(tag),
                "category": "cash_flow"
            }

        # Add specific mappings for tags that need special handling
        mappings["RevenueFromContractWithCustomerExcludingAssessedTax"] = {
            "standard_name": "revenue",
            "category": "income_statement"
        }

        mappings["SalesRevenueNet"] = {
            "standard_name": "revenue",
            "category": "income_statement"
        }

        mappings["CostOfGoodsAndServicesSold"] = {
            "standard_name": "cost_of_revenue",
            "category": "income_statement"
        }

        return mappings

    def _normalize_tag(self, tag: str) -> str:
        """Convert XBRL tag to normalized metric name.

        Args:
            tag: XBRL tag

        Returns:
            Normalized metric name
        """
        # Convert camel case to snake case
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', tag)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

        # Remove common prefixes/suffixes
        s3 = re.sub(r'(^us_gaap_|_member$)', '', s2)

        return s3

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

            # Get filing URL if not provided
            if not filing_url:
                # Use edgar API to get the filing URL
                try:
                    # Get the entity data
                    entity = edgar.get_entity(ticker)

                    # Get the filings
                    filings = entity.get_filings()

                    # Find the filing with the matching accession number
                    for filing in filings:
                        if filing.accession_number == accession_number:
                            # Construct the filing URL
                            cik = entity.cik
                            accession_clean = accession_number.replace('-', '')
                            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{accession_number}-index.htm"
                            break

                    if not filing_url:
                        raise ValueError(f"Filing with accession number {accession_number} not found")
                except Exception as e:
                    logger.error(f"Error getting filing URL: {e}")
                    return {
                        "filing_id": filing_id,
                        "ticker": ticker,
                        "accession_number": accession_number,
                        "error": f"Failed to get filing URL: {str(e)}"
                    }

            logger.info(f"Extracting XBRL data for {ticker} {accession_number} from {filing_url}")

            # Get XBRL data using the edgar package
            try:
                xbrl_data = XBRLData.from_filing(filing_url)
            except Exception as e:
                logger.error(f"Error loading XBRL data: {e}")
                return {
                    "filing_id": filing_id,
                    "ticker": ticker,
                    "accession_number": accession_number,
                    "error": f"Failed to load XBRL data: {str(e)}"
                }

            # Extract filing metadata
            filing_date = xbrl_data.period_end
            filing_type = self._extract_filing_type(filing_url)
            fiscal_year = int(filing_date.split('-')[0]) if filing_date and '-' in filing_date else None
            fiscal_quarter = self._determine_fiscal_quarter(filing_date, filing_type)

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
                "metrics": {}
            }

            # Extract financial statements
            self._extract_financial_statements(xbrl_data, financials)

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

    def _extract_filing_type(self, filing_url: str) -> Optional[str]:
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

            # If not found in URL, try to parse the filing content
            # This would require more complex logic and is not implemented here

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

    def _extract_financial_statements(self, xbrl_data: XBRLData, financials: Dict[str, Any]) -> None:
        """Extract financial statements from XBRL data.

        Args:
            xbrl_data: XBRL data object
            financials: Dictionary to store financial data
        """
        try:
            # Get available statements
            statements = xbrl_data.statements

            # Process each statement
            for statement_name, statement in statements.items():
                try:
                    # Skip non-financial statements
                    if not self._is_financial_statement(statement_name):
                        continue

                    # Get statement category
                    category = self._get_statement_category(statement_name)

                    # Process statement data
                    self._process_statement(statement, category, financials)

                except Exception as e:
                    logger.warning(f"Error processing statement {statement_name}: {e}")
        except Exception as e:
            logger.warning(f"Error extracting financial statements: {e}")

    def _is_financial_statement(self, statement_name: str) -> bool:
        """Check if a statement is a financial statement.

        Args:
            statement_name: Name of the statement

        Returns:
            True if it's a financial statement, False otherwise
        """
        # Common financial statement names
        financial_statements = [
            'income', 'statement of income', 'statement of operations', 'operations',
            'balance', 'balance sheet', 'statement of financial position', 'financial position',
            'cash flow', 'statement of cash flows', 'cash flows',
            'equity', 'statement of equity', 'stockholders equity', 'shareholders equity',
            'comprehensive income', 'statement of comprehensive income'
        ]

        # Check if any of the financial statement names is in the statement name
        statement_name_lower = statement_name.lower()
        return any(fs in statement_name_lower for fs in financial_statements)

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

    def _process_statement(self, statement: Any, category: str, financials: Dict[str, Any]) -> None:
        """Process a financial statement.

        Args:
            statement: Statement object
            category: Statement category
            financials: Dictionary to store financial data
        """
        try:
            # Get statement data as a pandas DataFrame
            df = statement.to_pandas()

            # Skip empty statements
            if df.empty:
                return

            # Process each row in the statement
            for _, row in df.iterrows():
                try:
                    # Skip rows without a concept
                    if 'concept' not in row or not row['concept']:
                        continue

                    # Get concept name and value
                    concept = row['concept']
                    value = None

                    # Try to get the value from different columns
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
                    standard_name = self._normalize_tag(concept)

                    # Get standard mapping if available
                    mapping = self.tag_mappings.get(concept, {})
                    if mapping:
                        standard_name = mapping.get('standard_name', standard_name)

                    # Create fact entry
                    fact = {
                        "xbrl_tag": concept,
                        "metric_name": standard_name,
                        "value": value,
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
