"""
Edgar XBRL to DuckDB Extractor

This module provides functionality to extract financial data from SEC filings
using the edgar library's XBRL parsing capabilities and store it in a DuckDB database.
"""

import hashlib
import logging
from typing import Any, Dict, Optional

import edgar
import pandas as pd
from edgar.financials import Financials
from edgar.xbrl.xbrldata import XBRLData

from ..storage.optimized_duckdb_store import OptimizedDuckDBStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgarXBRLToDuckDBExtractor:
    """
    A class to extract financial data from SEC filings using the edgar library's
    XBRL parsing capabilities and store it in a DuckDB database.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        batch_size: int = 100,
        read_only: bool = True,
    ):
        """
        Initialize the XBRL to DuckDB extractor.

        Args:
            db_path: Path to the DuckDB database file
            batch_size: Size of batches for bulk operations
            read_only: Whether to open the database in read-only mode
        """
        self.db = OptimizedDuckDBStore(
            db_path=db_path, batch_size=batch_size, read_only=read_only
        )
        logger.info(
            f"Initialized Edgar XBRL to DuckDB extractor with database at {self.db.db_path} (read_only={read_only})"
        )

    def process_filing(self, ticker: str, accession_number: str) -> Dict[str, Any]:
        """
        Process a filing and store its financial data in the DuckDB database.

        Args:
            ticker: Company ticker symbol
            accession_number: SEC accession number

        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing filing {ticker} {accession_number}")

            # Get the entity
            entity = edgar.get_entity(ticker)
            if not entity:
                logger.error(f"Entity not found: {ticker}")
                return {"error": f"Entity not found: {ticker}"}

            # Store company information
            self._store_company_info(entity, ticker)

            # Find the filing with the matching accession number
            filing = None
            for f in entity.get_filings():
                if f.accession_number == accession_number:
                    filing = f
                    break

            if not filing:
                logger.error(f"Filing not found: {ticker} {accession_number}")
                return {"error": f"Filing not found: {accession_number}"}

            # Check if the filing has XBRL data
            has_xbrl = hasattr(filing, "is_xbrl") and filing.is_xbrl

            # Generate a unique filing ID
            filing_id = f"{ticker}_{accession_number}"

            # Extract filing metadata
            filing_data = {
                "id": filing_id,
                "ticker": ticker,
                "accession_number": accession_number,
                "filing_type": filing.form,
                "filing_date": filing.filing_date,
                "document_url": filing.filing_url,
                "has_xbrl": has_xbrl,
            }

            # Store filing information
            self.db.store_filing(filing_data)

            # If the filing doesn't have XBRL data, we're done
            if not has_xbrl:
                logger.warning(f"Filing {accession_number} does not have XBRL data")
                return {
                    "status": "success",
                    "message": "Filing processed but no XBRL data available",
                    "filing_id": filing_id,
                    "has_xbrl": False,
                }

            # Extract XBRL data
            logger.info(f"Extracting XBRL data for {ticker} {accession_number}")

            # Use the Financials class for high-level access to financial statements
            financials = Financials.extract(filing)

            if not financials:
                logger.warning(
                    f"Could not extract financials for {ticker} {accession_number}"
                )
                return {
                    "status": "success",
                    "message": "Filing processed but could not extract financials",
                    "filing_id": filing_id,
                    "has_xbrl": True,
                }

            # Extract fiscal period information from DEI facts
            xbrl_data = filing.xbrl()
            fiscal_info = self._extract_fiscal_info(xbrl_data)

            # Update filing with fiscal information
            if fiscal_info:
                filing_data.update(fiscal_info)
                self.db.update_filing(filing_data)

            # Process financial statements
            self._process_financial_statements(
                financials, filing_id, ticker, fiscal_info
            )

            # Process US-GAAP facts
            self._process_us_gaap_facts(xbrl_data, filing_id, ticker, fiscal_info)

            return {
                "status": "success",
                "message": "Filing processed successfully",
                "filing_id": filing_id,
                "has_xbrl": True,
                "fiscal_info": fiscal_info,
            }

        except Exception as e:
            logger.error(f"Error processing filing {ticker} {accession_number}: {e}")
            return {"error": str(e)}

    def _store_company_info(self, entity, ticker: str):
        """
        Store company information in the database.

        Args:
            entity: Edgar entity object
            ticker: Company ticker symbol
        """
        try:
            company_data = {
                "ticker": ticker,
                "name": entity.name,
                "cik": entity.cik,
                "sic": getattr(entity, "sic", None),
                "sector": getattr(entity, "sector", None),
                "industry": getattr(entity, "industry", None),
                "exchange": getattr(entity, "exchange", None),
            }
            self.db.store_company(company_data)
        except Exception as e:
            logger.warning(f"Error storing company info for {ticker}: {e}")

    def _extract_fiscal_info(self, xbrl_data: XBRLData) -> Dict[str, Any]:
        """
        Extract fiscal period information from XBRL data.

        Args:
            xbrl_data: XBRL data object

        Returns:
            Dictionary with fiscal period information
        """
        fiscal_info = {}

        try:
            # Get fiscal year and period from DEI facts
            if hasattr(xbrl_data, "instance"):
                fiscal_year = xbrl_data.instance.get_fiscal_year_focus()
                fiscal_period = xbrl_data.instance.get_fiscal_period_focus()
                period_end_date = xbrl_data.instance.get_document_period()

                if fiscal_year:
                    fiscal_info["fiscal_year"] = int(fiscal_year)

                if fiscal_period:
                    # Convert fiscal period (e.g., Q1, Q2, Q3, FY) to quarter number
                    quarter_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}
                    fiscal_info["fiscal_quarter"] = quarter_map.get(fiscal_period, None)

                if period_end_date:
                    fiscal_info["fiscal_period_end_date"] = period_end_date
        except Exception as e:
            logger.warning(f"Error extracting fiscal info: {e}")

        return fiscal_info

    def _process_financial_statements(
        self,
        financials: Financials,
        filing_id: str,
        ticker: str,
        fiscal_info: Dict[str, Any],
    ):
        """
        Process financial statements and store them in the database.

        Args:
            financials: Financials object
            filing_id: Filing ID
            ticker: Company ticker symbol
            fiscal_info: Fiscal period information
        """
        try:
            # Get standard financial statements
            balance_sheet = financials.get_balance_sheet()
            income_statement = financials.get_income_statement()
            cash_flow = financials.get_cash_flow_statement()

            # Process balance sheet
            if balance_sheet:
                self._process_statement(
                    balance_sheet, "balance_sheet", filing_id, ticker, fiscal_info
                )

            # Process income statement
            if income_statement:
                self._process_statement(
                    income_statement, "income_statement", filing_id, ticker, fiscal_info
                )

            # Process cash flow statement
            if cash_flow:
                self._process_statement(
                    cash_flow, "cash_flow", filing_id, ticker, fiscal_info
                )

        except Exception as e:
            logger.warning(f"Error processing financial statements: {e}")

    def _process_statement(
        self,
        statement,
        statement_type: str,
        filing_id: str,
        ticker: str,
        fiscal_info: Dict[str, Any],
    ):
        """
        Process a financial statement and store its data in the database.

        Args:
            statement: Statement object
            statement_type: Type of statement (balance_sheet, income_statement, cash_flow)
            filing_id: Filing ID
            ticker: Company ticker symbol
            fiscal_info: Fiscal period information
        """
        try:
            # Convert statement to DataFrame
            if hasattr(statement, "data") and isinstance(statement.data, pd.DataFrame):
                df = statement.data.reset_index()

                # Process each row in the DataFrame
                facts = []
                metrics = []

                for _, row in df.iterrows():
                    concept = row.get("concept")
                    if not concept:
                        continue

                    # Get the label
                    label = row.get("label", self._normalize_concept_name(concept))

                    # Process each period column
                    for col in df.columns:
                        if col not in ["concept", "label", "level"]:
                            value = row.get(col)
                            if pd.notna(value):
                                # Try to convert to float
                                try:
                                    value = float(value)
                                except (ValueError, TypeError):
                                    continue

                                # Generate a unique ID for the fact
                                fact_id = self._generate_id(
                                    f"{filing_id}_{concept}_{col}"
                                )

                                # Parse the period information
                                period_info = self._parse_period_info(col)

                                # Create fact entry
                                fact = {
                                    "id": fact_id,
                                    "filing_id": filing_id,
                                    "xbrl_tag": concept,
                                    "metric_name": label,
                                    "value": value,
                                    "unit": "USD",  # Assuming USD for financial statements
                                    "period_type": period_info.get(
                                        "period_type", "duration"
                                    ),
                                    "start_date": period_info.get("start_date"),
                                    "end_date": period_info.get("end_date"),
                                    "segment": statement_type,
                                    "context_id": None,
                                    "decimals": None,
                                }
                                facts.append(fact)

                                # Create time series metric entry
                                if fiscal_info and "fiscal_year" in fiscal_info:
                                    metric = {
                                        "ticker": ticker,
                                        "metric_name": label,
                                        "fiscal_year": fiscal_info.get("fiscal_year"),
                                        "fiscal_quarter": fiscal_info.get(
                                            "fiscal_quarter"
                                        ),
                                        "value": value,
                                        "unit": "USD",
                                        "filing_id": filing_id,
                                        "statement_type": statement_type,
                                    }
                                    metrics.append(metric)

                # Store facts and metrics in batches
                if facts:
                    self.db.store_financial_facts_batch(facts)

                if metrics:
                    self.db.store_time_series_metrics_batch(metrics)

                logger.info(
                    f"Processed {statement_type} with {len(facts)} facts and {len(metrics)} metrics"
                )

        except Exception as e:
            logger.warning(f"Error processing {statement_type}: {e}")

    def _process_us_gaap_facts(
        self,
        xbrl_data: XBRLData,
        filing_id: str,
        ticker: str,
        fiscal_info: Dict[str, Any],
    ):
        """
        Process US-GAAP facts and store them in the database.

        Args:
            xbrl_data: XBRL data object
            filing_id: Filing ID
            ticker: Company ticker symbol
            fiscal_info: Fiscal period information
        """
        try:
            if not hasattr(xbrl_data, "instance"):
                logger.warning("XBRL data does not have instance attribute")
                return

            # Query all US-GAAP facts
            us_gaap_facts = xbrl_data.instance.query_facts(schema="us-gaap")

            if us_gaap_facts.empty:
                logger.warning("No US-GAAP facts found")
                return

            # Process facts
            facts = []
            metrics = []

            for _, row in us_gaap_facts.iterrows():
                concept = row.get("concept")
                value = row.get("value")

                # Skip if no value
                if pd.isna(value):
                    continue

                # Try to convert to float
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue

                # Generate a unique ID for the fact
                fact_id = self._generate_id(
                    f"{filing_id}_{concept}_{row.get('context_id')}"
                )

                # Create fact entry
                fact = {
                    "id": fact_id,
                    "filing_id": filing_id,
                    "xbrl_tag": concept,
                    "metric_name": self._normalize_concept_name(concept),
                    "value": value,
                    "unit": row.get("units"),
                    "period_type": row.get("period_type"),
                    "start_date": row.get("start_date"),
                    "end_date": row.get("end_date"),
                    "segment": None,
                    "context_id": row.get("context_id"),
                    "decimals": row.get("decimals"),
                }
                facts.append(fact)

                # Create time series metric entry
                if fiscal_info and "fiscal_year" in fiscal_info:
                    metric = {
                        "ticker": ticker,
                        "metric_name": self._normalize_concept_name(concept),
                        "fiscal_year": fiscal_info.get("fiscal_year"),
                        "fiscal_quarter": fiscal_info.get("fiscal_quarter"),
                        "value": value,
                        "unit": row.get("units"),
                        "filing_id": filing_id,
                        "statement_type": "us_gaap_fact",
                    }
                    metrics.append(metric)

            # Store facts and metrics in batches
            if facts:
                self.db.store_financial_facts_batch(facts)

            if metrics:
                self.db.store_time_series_metrics_batch(metrics)

            logger.info(
                f"Processed {len(facts)} US-GAAP facts and {len(metrics)} metrics"
            )

        except Exception as e:
            logger.warning(f"Error processing US-GAAP facts: {e}")

    def _normalize_concept_name(self, concept: str) -> str:
        """
        Normalize concept name by removing namespace prefix and converting to snake_case.

        Args:
            concept: XBRL concept name

        Returns:
            Normalized concept name
        """
        # Remove namespace prefix if present
        if ":" in concept:
            concept = concept.split(":")[1]

        # Convert camel case to snake case
        result = ""
        for i, char in enumerate(concept):
            if i > 0 and char.isupper() and concept[i - 1].islower():
                result += "_"
            result += char.lower()

        return result

    def _parse_period_info(self, period_str: str) -> Dict[str, Any]:
        """
        Parse period information from a period string.

        Args:
            period_str: Period string (e.g., '2021-12-31', '2021-01-01 to 2021-12-31')

        Returns:
            Dictionary with period information
        """
        period_info = {}

        try:
            if " to " in period_str:
                start_date, end_date = period_str.split(" to ")
                period_info["period_type"] = "duration"
                period_info["start_date"] = start_date
                period_info["end_date"] = end_date
            else:
                period_info["period_type"] = "instant"
                period_info["end_date"] = period_str
        except Exception:
            pass

        return period_info

    def _generate_id(self, input_str: str) -> str:
        """
        Generate a unique ID from an input string.

        Args:
            input_str: Input string

        Returns:
            Unique ID
        """
        return hashlib.md5(input_str.encode()).hexdigest()
