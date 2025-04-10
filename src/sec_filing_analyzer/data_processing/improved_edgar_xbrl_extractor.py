"""
Improved Edgar XBRL to DuckDB Extractor

A module for extracting financial data from SEC filings using the edgar library's
XBRL parsing capabilities and storing it in a DuckDB database using the improved schema.
"""

import logging
import hashlib
import pandas as pd
from typing import Dict, List, Optional, Any
from edgar import Filing, Financials, XBRLData

from sec_filing_analyzer.utils import edgar_utils

from sec_filing_analyzer.storage.improved_duckdb_store import ImprovedDuckDBStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ImprovedEdgarXBRLExtractor:
    """
    A class to extract financial data from SEC filings using the edgar library's
    XBRL parsing capabilities and store it in a DuckDB database using the improved schema.
    """

    def __init__(self, db_path: Optional[str] = None, batch_size: int = 100):
        """
        Initialize the XBRL to DuckDB extractor.

        Args:
            db_path: Path to the DuckDB database file
            batch_size: Size of batches for bulk operations
        """
        self.db = ImprovedDuckDBStore(db_path=db_path)
        self.batch_size = batch_size
        logger.info(f"Initialized Improved Edgar XBRL extractor with database at {self.db.db_path}")

    def _generate_id(self, text: str) -> str:
        """
        Generate a unique ID from text.

        Args:
            text: Text to hash

        Returns:
            Hashed text
        """
        return hashlib.md5(text.encode()).hexdigest()

    def _normalize_concept_name(self, concept: str) -> str:
        """
        Normalize XBRL concept name to a standard metric name.

        Args:
            concept: XBRL concept name

        Returns:
            Normalized metric name
        """
        # Remove namespace prefix if present
        if ':' in concept:
            concept = concept.split(':')[-1]

        # Convert camel case to snake case
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', concept)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

        return s2

    def _extract_fiscal_info(self, xbrl_data: XBRLData) -> Dict[str, Any]:
        """
        Extract fiscal period information from DEI facts.

        Args:
            xbrl_data: XBRL data object

        Returns:
            Dictionary containing fiscal period information
        """
        fiscal_info = {}

        try:
            if not hasattr(xbrl_data, 'instance'):
                logger.warning("XBRL data does not have instance attribute")
                return fiscal_info

            # Query DEI facts
            dei_facts = xbrl_data.instance.query_facts(schema='dei')

            if dei_facts.empty:
                logger.warning("No DEI facts found")
                return fiscal_info

            # Extract fiscal year
            fiscal_year_end = dei_facts[dei_facts['concept'].str.contains('FiscalYear', case=True, na=False)]
            if not fiscal_year_end.empty:
                fiscal_year = fiscal_year_end.iloc[0].get('value')
                try:
                    fiscal_info['fiscal_year'] = int(fiscal_year)
                except (ValueError, TypeError):
                    pass

            # Extract fiscal period
            fiscal_period = dei_facts[dei_facts['concept'].str.contains('FiscalPeriod', case=True, na=False)]
            if not fiscal_period.empty:
                period = fiscal_period.iloc[0].get('value')
                if period:
                    # Map fiscal period to quarter
                    if period.upper() in ['Q1', 'FQ1', '1Q']:
                        fiscal_info['fiscal_period'] = 'Q1'
                        fiscal_info['fiscal_quarter'] = 1
                    elif period.upper() in ['Q2', 'FQ2', '2Q']:
                        fiscal_info['fiscal_period'] = 'Q2'
                        fiscal_info['fiscal_quarter'] = 2
                    elif period.upper() in ['Q3', 'FQ3', '3Q']:
                        fiscal_info['fiscal_period'] = 'Q3'
                        fiscal_info['fiscal_quarter'] = 3
                    elif period.upper() in ['FY', 'Y', 'Q4', 'FQ4', '4Q']:
                        fiscal_info['fiscal_period'] = 'FY'
                        fiscal_info['fiscal_quarter'] = 4

            # Extract fiscal period end date
            period_end = dei_facts[dei_facts['concept'].str.contains('DocumentPeriodEndDate', case=True, na=False)]
            if not period_end.empty:
                end_date = period_end.iloc[0].get('value')
                if end_date:
                    fiscal_info['fiscal_period_end_date'] = end_date

        except Exception as e:
            logger.warning(f"Error extracting fiscal info: {e}")

        return fiscal_info

    def process_filing(self, ticker: str, accession_number: str) -> Dict[str, Any]:
        """
        Process a single SEC filing and store its data in the database.

        Args:
            ticker: Company ticker symbol
            accession_number: SEC accession number

        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info(f"Processing filing {ticker} {accession_number}")

            # Check if the filing already exists in the database
            if self.db.filing_exists(accession_number):
                logger.info(f"Filing {accession_number} already exists in the database")
                filing_id = self.db.get_filing_id(accession_number)
                return {
                    "status": "success",
                    "message": "Filing already exists in the database",
                    "filing_id": filing_id
                }

            # Get the filing using edgar_utils
            filing = edgar_utils.get_filing_by_accession(ticker, accession_number)

            if not filing:
                logger.error(f"Filing {accession_number} not found for {ticker}")
                return {"error": f"Filing {accession_number} not found for {ticker}"}

            # Check if the filing has XBRL data
            has_xbrl = hasattr(filing, 'is_xbrl') and filing.is_xbrl

            # Get filing metadata
            metadata = edgar_utils.get_filing_metadata(filing, ticker)

            # Store company information
            company_data = {
                "ticker": ticker,
                "name": metadata.get("company"),
                "cik": metadata.get("cik")
            }
            company_id = self.db.store_company(company_data)

            if not company_id:
                logger.error(f"Failed to store company {ticker}")
                return {"error": "Failed to store company"}

            # Extract filing metadata
            filing_data = {
                "company_id": company_id,
                "accession_number": accession_number,
                "filing_type": metadata.get("form"),
                "filing_date": metadata.get("filing_date"),
                "document_url": metadata.get("filing_url"),
                "has_xbrl": has_xbrl
            }

            # Store filing information
            filing_id = self.db.store_filing(filing_data)

            if not filing_id:
                logger.error(f"Failed to store filing {accession_number}")
                return {"error": "Failed to store filing"}

            # If the filing doesn't have XBRL data, we're done
            if not has_xbrl:
                logger.warning(f"Filing {accession_number} does not have XBRL data")
                return {
                    "status": "success",
                    "message": "Filing processed but no XBRL data available",
                    "filing_id": filing_id,
                    "has_xbrl": False
                }

            # Extract XBRL data
            logger.info(f"Extracting XBRL data for {ticker} {accession_number}")

            # Use the Financials class for high-level access to financial statements
            financials = Financials.extract(filing)

            if not financials:
                logger.warning(f"Could not extract financials for {ticker} {accession_number}")
                return {
                    "status": "success",
                    "message": "Filing processed but could not extract financials",
                    "filing_id": filing_id,
                    "has_xbrl": True
                }

            # Extract fiscal period information from DEI facts
            xbrl_data = filing.xbrl()
            fiscal_info = self._extract_fiscal_info(xbrl_data)

            # Update filing with fiscal information
            if fiscal_info:
                filing_data.update(fiscal_info)
                self.db.store_filing(filing_data)

            # Process financial statements
            self._process_financial_statements(financials, filing_id, fiscal_info)

            # Process US-GAAP facts
            self._process_us_gaap_facts(xbrl_data, filing_id)

            return {
                "status": "success",
                "message": "Filing processed successfully",
                "filing_id": filing_id,
                "has_xbrl": True,
                "fiscal_info": fiscal_info
            }

        except Exception as e:
            logger.error(f"Error processing filing {ticker} {accession_number}: {e}")
            return {"error": str(e)}

    def process_company(self, ticker: str, filing_types: Optional[List[str]] = None,
                       limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all filings for a company.

        Args:
            ticker: Company ticker symbol
            filing_types: List of filing types to process (e.g., ['10-K', '10-Q'])
            limit: Maximum number of filings to process

        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info(f"Processing filings for {ticker}")

            # Default to 10-K and 10-Q filings if not specified
            if filing_types is None:
                filing_types = ['10-K', '10-Q']

            # Get all filings for the company using edgar_utils
            filings = []
            for form_type in filing_types:
                form_filings = edgar_utils.get_filings(ticker, form_type=form_type, limit=limit)
                filings.extend(form_filings)

            if not filings:
                logger.warning(f"No filings found for {ticker}")
                return {"error": "No filings found"}

            # Limit the number of filings if specified
            if limit and len(filings) > limit:
                filings = filings[:limit]

            results = []
            for filing in filings:
                accession_number = filing.accession_number
                result = self.process_filing(ticker, accession_number)
                results.append(result)

            return {
                "status": "success",
                "message": f"Processed {len(results)} filings for {ticker}",
                "results": results
            }

        except Exception as e:
            logger.error(f"Error processing company {ticker}: {e}")
            return {"error": str(e)}

    def _process_financial_statements(self, financials: Financials, filing_id: int,
                                    fiscal_info: Dict[str, Any]):
        """
        Process financial statements and store their data in the database.

        Args:
            financials: Financials object
            filing_id: Filing ID
            fiscal_info: Fiscal period information
        """
        try:
            # Get standard financial statements
            balance_sheet = financials.get_balance_sheet()
            income_statement = financials.get_income_statement()
            cash_flow = financials.get_cash_flow_statement()

            # Process balance sheet
            if balance_sheet:
                self._process_statement(balance_sheet, "balance_sheet", filing_id, fiscal_info)

            # Process income statement
            if income_statement:
                self._process_statement(income_statement, "income_statement", filing_id, fiscal_info)

            # Process cash flow statement
            if cash_flow:
                self._process_statement(cash_flow, "cash_flow", filing_id, fiscal_info)

        except Exception as e:
            logger.warning(f"Error processing financial statements: {e}")

    def _process_statement(self, statement, statement_type: str, filing_id: int,
                          fiscal_info: Dict[str, Any]):
        """
        Process a financial statement and store its data in the database.

        Args:
            statement: Statement object
            statement_type: Type of statement (balance_sheet, income_statement, cash_flow)
            filing_id: Filing ID
            fiscal_info: Fiscal period information
        """
        try:
            # Convert statement to DataFrame
            if hasattr(statement, 'data') and isinstance(statement.data, pd.DataFrame):
                df = statement.data.reset_index()

                # Process each row in the DataFrame
                facts = []

                for _, row in df.iterrows():
                    # Skip rows without a label
                    if 'label' not in row or pd.isna(row['label']):
                        continue

                    label = row['label']

                    # Find the value column (usually the first numeric column)
                    value_col = None
                    value = None

                    for col in df.columns:
                        if col != 'label' and pd.api.types.is_numeric_dtype(df[col]):
                            value_col = col
                            value = row[col]
                            break

                    if value_col is None or pd.isna(value):
                        continue

                    # Normalize the label to create a metric name
                    metric_name = self._normalize_concept_name(label)

                    # Store the metric definition
                    metric_data = {
                        "metric_name": metric_name,
                        "display_name": label,
                        "category": statement_type,
                        "unit_of_measure": "USD"  # Assuming financial values are in USD
                    }
                    metric_id = self.db.store_metric(metric_data)

                    if not metric_id:
                        logger.warning(f"Failed to store metric {metric_name}")
                        continue

                    # Create fact entry
                    fact = {
                        "filing_id": filing_id,
                        "metric_id": metric_id,
                        "value": value,
                        "as_reported": True,
                        "period_type": "duration",  # Assuming most financial statement items are for a period
                        "end_date": fiscal_info.get("fiscal_period_end_date"),
                        "context_id": f"{statement_type}_{label}"
                    }
                    facts.append(fact)

                # Store facts in batches
                if facts:
                    self.db.store_facts_batch(facts)

                logger.info(f"Processed {statement_type} with {len(facts)} facts")

        except Exception as e:
            logger.warning(f"Error processing {statement_type}: {e}")

    def _process_us_gaap_facts(self, xbrl_data: XBRLData, filing_id: int):
        """
        Process US-GAAP facts and store them in the database.

        Args:
            xbrl_data: XBRL data object
            filing_id: Filing ID
            fiscal_info: Fiscal period information
        """
        try:
            if not hasattr(xbrl_data, 'instance'):
                logger.warning(f"XBRL data does not have instance attribute")
                return

            # Query all US-GAAP facts
            us_gaap_facts = xbrl_data.instance.query_facts(schema='us-gaap')

            if us_gaap_facts.empty:
                logger.warning(f"No US-GAAP facts found")
                return

            # Process facts
            facts = []

            for _, row in us_gaap_facts.iterrows():
                concept = row.get('concept')
                value = row.get('value')

                # Skip if no value
                if pd.isna(value):
                    continue

                # Try to convert to float
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue

                # Normalize the concept to create a metric name
                metric_name = self._normalize_concept_name(concept)

                # Store the metric definition
                metric_data = {
                    "metric_name": metric_name,
                    "display_name": concept,
                    "category": "us_gaap",  # Default category for US-GAAP facts
                    "unit_of_measure": row.get('units', 'USD')  # Assuming financial values are in USD
                }
                metric_id = self.db.store_metric(metric_data)

                if not metric_id:
                    logger.warning(f"Failed to store metric {metric_name}")
                    continue

                # Create fact entry
                fact = {
                    "filing_id": filing_id,
                    "metric_id": metric_id,
                    "value": value,
                    "as_reported": True,
                    "period_type": row.get('period_type'),
                    "start_date": row.get('start_date'),
                    "end_date": row.get('end_date'),
                    "context_id": row.get('context_id'),
                    "decimals": row.get('decimals')
                }
                facts.append(fact)

            # Store facts in batches
            if facts:
                self.db.store_facts_batch(facts)

            logger.info(f"Processed {len(facts)} US-GAAP facts")

        except Exception as e:
            logger.warning(f"Error processing US-GAAP facts: {e}")

    def close(self):
        """Close the database connection."""
        if hasattr(self, 'db'):
            self.db.close()

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()
