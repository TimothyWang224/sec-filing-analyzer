"""
Parallel XBRL Extractor Module

This module provides a parallel implementation for extracting financial data from XBRL filings
using the edgartools library (aliased as edgar).
"""

import logging
import json
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Import edgartools components (aliased as edgar)
import edgar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelXBRLExtractor:
    """
    A parallel extractor for XBRL data from SEC filings using the edgar package.
    """

    def __init__(self, cache_dir: Optional[str] = None, max_workers: int = 4,
                rate_limit: float = 0.2):
        """Initialize the parallel XBRL extractor.

        Args:
            cache_dir: Optional directory to cache XBRL data
            max_workers: Maximum number of worker threads
            rate_limit: Minimum time between API requests in seconds
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/xbrl_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.last_request_time = 0

        # Precompile regex patterns for text extraction
        import re
        self.text_patterns = {
            "revenue": re.compile(r"(?:Total|Net)\s+[Rr]evenue[s]?\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$", re.MULTILINE),
            "net_income": re.compile(r"(?:Net|Total)\s+[Ii]ncome\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$", re.MULTILINE),
            "total_assets": re.compile(r"(?:Total|All)\s+[Aa]ssets\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$", re.MULTILINE),
            "total_liabilities": re.compile(r"(?:Total|All)\s+[Ll]iabilities\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$", re.MULTILINE),
            "stockholders_equity": re.compile(r"(?:Total|All)\s+(?:[Ss]tockholders'?|[Ss]hareholders'?)\s+[Ee]quity\s*[:\-]?\s*[$]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|thousand|\$)?\s*$", re.MULTILINE)
        }

        logger.info(f"Initialized parallel XBRL extractor with cache at {self.cache_dir} and {max_workers} workers")

    def _apply_rate_limit(self):
        """Apply rate limiting to API requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        self.last_request_time = time.time()

    def extract_financials(self, ticker: str, filing_id: str,
                          accession_number: str, filing_url: Optional[str] = None) -> Dict[str, Any]:
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

            # Apply rate limiting
            self._apply_rate_limit()

            # Get XBRL data using the edgar package
            try:
                # Get the entity
                entity = edgar.get_entity(ticker)

                # Get the filings
                filings = entity.get_filings()

                # Find the filing with the matching accession number
                filing_obj = None
                for filing in filings:
                    if filing.accession_number == accession_number:
                        filing_obj = filing
                        break

                if not filing_obj:
                    raise ValueError(f"Filing with accession number {accession_number} not found")

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
                    "filing_url": filing_url or str(filing_obj.filing_url) if hasattr(filing_obj, 'filing_url') else None,
                    "filing_date": filing_date,
                    "fiscal_year": fiscal_year,
                    "fiscal_quarter": fiscal_quarter,
                    "filing_type": filing_type,
                    "facts": [],
                    "metrics": {},
                    "statements": {}
                }

                # Extract financial statements
                self._extract_financial_data(filing_obj, financials)

                # Calculate key financial ratios
                self._calculate_ratios(financials)

                # Cache the results
                with open(cache_file, 'w') as f:
                    json.dump(financials, f, indent=2)

                logger.info(f"Extracted {len(financials['facts'])} facts for {ticker} {accession_number}")
                return financials

            except Exception as e:
                logger.error(f"Error extracting XBRL data: {e}")
                return {
                    "filing_id": filing_id,
                    "ticker": ticker,
                    "accession_number": accession_number,
                    "error": f"Failed to extract XBRL data: {str(e)}"
                }

        except Exception as e:
            logger.error(f"Error extracting XBRL data for {ticker} {accession_number}: {str(e)}")
            return {
                "filing_id": filing_id,
                "ticker": ticker,
                "accession_number": accession_number,
                "error": str(e)
            }

    def extract_financials_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract financial data for a batch of filings.

        Args:
            batch: List of dictionaries with ticker, filing_id, and accession_number

        Returns:
            List of financial data dictionaries
        """
        results = []

        for item in batch:
            ticker = item.get("ticker")
            filing_id = item.get("filing_id")
            accession_number = item.get("accession_number")
            filing_url = item.get("filing_url")

            if not ticker or not filing_id or not accession_number:
                logger.warning(f"Missing required parameters in batch item: {item}")
                continue

            result = self.extract_financials(
                ticker=ticker,
                filing_id=filing_id,
                accession_number=accession_number,
                filing_url=filing_url
            )

            results.append(result)

        return results

    def extract_financials_for_companies(self, companies: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract financial data for multiple companies in parallel.

        Args:
            companies: List of dictionaries with ticker and filings list
                Each filing should have filing_id and accession_number

        Returns:
            Dictionary mapping tickers to lists of financial data
        """
        results = {}
        all_filings = []

        # Prepare all filings from all companies
        for company in companies:
            ticker = company.get("ticker")
            filings = company.get("filings", [])

            if not ticker or not filings:
                continue

            for filing in filings:
                filing["ticker"] = ticker
                all_filings.append(filing)

        # Process filings in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Split into batches to avoid overwhelming the executor
            batch_size = 10
            batches = [all_filings[i:i + batch_size] for i in range(0, len(all_filings), batch_size)]

            # Submit batch tasks
            future_to_batch = {
                executor.submit(self.extract_financials_batch, batch): i
                for i, batch in enumerate(batches)
            }

            # Process results
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_results = future.result()

                # Organize results by ticker
                for result in batch_results:
                    ticker = result.get("ticker")

                    if ticker not in results:
                        results[ticker] = []

                    results[ticker].append(result)

        return results

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

    def _extract_financial_data(self, filing_obj: Any, financials: Dict[str, Any]) -> None:
        """Extract financial data from filing.

        Args:
            filing_obj: Filing object
            financials: Dictionary to store financial data
        """
        try:
            # Extract statements if available
            self._extract_statements(filing_obj, financials)

            # Extract XBRL facts if available
            self._extract_xbrl_facts(filing_obj, financials)

            # If no data extracted, try text extraction
            if not financials["facts"]:
                self._extract_data_from_text(filing_obj, financials)
        except Exception as e:
            logger.warning(f"Error extracting financial data: {e}")

    def _extract_statements(self, filing_obj: Any, financials: Dict[str, Any]) -> None:
        """Extract financial statements from filing.

        Args:
            filing_obj: Filing object
            financials: Dictionary to store financial data
        """
        try:
            # Check if filing has statements
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
        except Exception as e:
            logger.warning(f"Error extracting statements: {e}")

    def _extract_xbrl_facts(self, filing_obj: Any, financials: Dict[str, Any]) -> None:
        """Extract XBRL facts from filing using edgar's XBRL capabilities.

        Args:
            filing_obj: Filing object
            financials: Dictionary to store financial data
        """
        try:
            # Check if filing has XBRL data
            if not hasattr(filing_obj, 'xbrl') or not filing_obj.xbrl:
                logger.info(f"Filing does not have XBRL data or xbrl attribute")
                return

            xbrl_data = filing_obj.xbrl

            # Extract facts directly from XBRL data
            self._extract_facts_from_xbrl(xbrl_data, financials)

            # Extract data from statements if available
            if hasattr(xbrl_data, 'statements') and xbrl_data.statements:
                self._extract_data_from_statements(xbrl_data.statements, financials)

            # Extract data from calculations if available
            if hasattr(xbrl_data, 'calculations') and xbrl_data.calculations:
                self._extract_data_from_calculations(xbrl_data.calculations, financials)

        except Exception as e:
            logger.warning(f"Error extracting XBRL facts: {e}")

    def _extract_facts_from_xbrl(self, xbrl_data: Any, financials: Dict[str, Any]) -> None:
        """Extract facts directly from XBRL data.

        Args:
            xbrl_data: XBRL data object
            financials: Dictionary to store financial data
        """
        try:
            # Check if XBRL data has facts
            if not hasattr(xbrl_data, 'facts') or not xbrl_data.facts:
                logger.info("XBRL data does not have facts")
                return

            facts = xbrl_data.facts

            # Process each fact
            for fact_id, fact in facts.items():
                try:
                    # Skip non-financial facts
                    if not self._is_financial_fact(fact_id):
                        continue

                    # Get fact value
                    value = self._get_fact_value(fact)
                    if value is None:
                        continue

                    # Get fact metadata
                    metric_name = self._normalize_fact_name(fact_id)

                    # Create fact entry
                    fact_entry = {
                        "xbrl_tag": fact_id,
                        "metric_name": metric_name,
                        "value": value,
                        "category": self._get_fact_category(fact_id)
                    }

                    # Add context information if available
                    if hasattr(fact, 'context') and fact.context:
                        context = fact.context
                        fact_entry.update(self._extract_context_info(context))

                    # Add to facts list
                    financials["facts"].append(fact_entry)

                    # Add to metrics dictionary if it's a primary fact
                    if self._is_primary_fact(fact_id, fact_entry.get("period_type")):
                        financials["metrics"][metric_name] = value

                except Exception as e:
                    logger.debug(f"Error processing fact {fact_id}: {e}")
        except Exception as e:
            logger.warning(f"Error extracting facts from XBRL: {e}")

    def _extract_data_from_calculations(self, calculations: Any, financials: Dict[str, Any]) -> None:
        """Extract data from XBRL calculations.

        Args:
            calculations: XBRL calculations
            financials: Dictionary to store financial data
        """
        try:
            # Process calculations to extract relationships between metrics
            for calc_id, calc in calculations.items():
                try:
                    # Skip non-financial calculations
                    if not self._is_financial_fact(calc_id):
                        continue

                    # Get calculation target
                    target_name = self._normalize_fact_name(calc_id)

                    # Process calculation components
                    if hasattr(calc, 'components') and calc.components:
                        for component in calc.components:
                            try:
                                # Get component details
                                if hasattr(component, 'concept') and component.concept:
                                    component_id = component.concept
                                    component_name = self._normalize_fact_name(component_id)

                                    # Add relationship to financials
                                    if "relationships" not in financials:
                                        financials["relationships"] = []

                                    relationship = {
                                        "source": component_name,
                                        "target": target_name,
                                        "type": "calculation",
                                        "weight": component.weight if hasattr(component, 'weight') else 1.0
                                    }

                                    financials["relationships"].append(relationship)
                            except Exception as e:
                                logger.debug(f"Error processing calculation component: {e}")
                except Exception as e:
                    logger.debug(f"Error processing calculation {calc_id}: {e}")
        except Exception as e:
            logger.warning(f"Error extracting data from calculations: {e}")

    def _extract_data_from_statements(self, statements: Any, financials: Dict[str, Any]) -> None:
        """Extract data from XBRL statements.

        Args:
            statements: XBRL statements
            financials: Dictionary to store financial data
        """
        try:
            # Process each statement
            for statement_name, statement in statements.items():
                try:
                    # Get statement category for metadata
                    category = self._get_statement_category(statement_name)

                    # Store category in statement metadata
                    statement_metadata = {"category": category, "name": statement_name}

                    # Convert statement to pandas DataFrame if possible
                    if hasattr(statement, 'to_pandas'):
                        df = statement.to_pandas()
                    elif isinstance(statement, dict):
                        df = pd.DataFrame(statement)
                    else:
                        # Skip if we can't convert to DataFrame
                        continue

                    # Skip empty statements
                    if df.empty:
                        continue

                    # Add statement to financials
                    if "statements" not in financials:
                        financials["statements"] = {}

                    # Store statement data with metadata
                    statement_data = {
                        "data": df.to_dict(orient='records'),
                        "metadata": statement_metadata
                    }
                    financials["statements"][statement_name] = statement_data

                    # Process statement data
                    self._process_statement_data(df, statement_name, financials)

                except Exception as e:
                    logger.warning(f"Error processing statement {statement_name}: {e}")
        except Exception as e:
            logger.warning(f"Error extracting data from statements: {e}")

    def _extract_data_from_text(self, filing_obj: Any, financials: Dict[str, Any]) -> None:
        """Extract financial data from filing text using edgar's capabilities.

        Args:
            filing_obj: Filing object
            financials: Dictionary to store financial data
        """
        try:
            # Try to use edgar's document parsing capabilities first
            if hasattr(filing_obj, 'document') and filing_obj.document:
                self._extract_from_document(filing_obj.document, financials)

                # If we extracted data, return
                if financials["facts"]:
                    return

            # Fallback to text extraction if document parsing failed
            # Get the filing text
            text = None
            if hasattr(filing_obj, 'text'):
                if callable(filing_obj.text):
                    try:
                        text = filing_obj.text()
                    except Exception as e:
                        logger.warning(f"Error calling text method: {e}")
                else:
                    text = filing_obj.text

            if not text:
                return

            # Extract metrics using regex patterns
            for metric_name, pattern in self.text_patterns.items():
                matches = pattern.findall(text)
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

    def _extract_from_document(self, document: Any, financials: Dict[str, Any]) -> None:
        """Extract financial data from edgar document.

        Args:
            document: Edgar document object
            financials: Dictionary to store financial data
        """
        try:
            # Try to extract tables from the document
            if hasattr(document, 'extract_tables') and callable(document.extract_tables):
                tables = document.extract_tables()

                if not tables:
                    return

                # Process each table
                for i, table in enumerate(tables):
                    try:
                        # Convert table to DataFrame
                        df = pd.DataFrame(table)

                        # Skip empty tables
                        if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
                            continue

                        # Try to identify financial tables
                        if self._is_financial_table(df):
                            # Process financial table
                            self._process_financial_table(df, f"table_{i}", financials)
                    except Exception as e:
                        logger.debug(f"Error processing table {i}: {e}")
        except Exception as e:
            logger.warning(f"Error extracting from document: {e}")

    def _is_financial_table(self, df: pd.DataFrame) -> bool:
        """Check if a table is a financial table.

        Args:
            df: DataFrame containing table data

        Returns:
            True if it's a financial table, False otherwise
        """
        # Check if the table has financial keywords in the header
        financial_keywords = [
            'revenue', 'income', 'earnings', 'profit', 'loss', 'assets', 'liabilities',
            'equity', 'cash', 'balance', 'statement', 'financial', 'fiscal', 'quarter',
            'annual', 'year', 'ended', 'consolidated'
        ]

        # Check first row and first column for financial keywords
        header_text = ' '.join(str(x).lower() for x in df.iloc[0].values if x is not None)
        col_text = ' '.join(str(x).lower() for x in df.iloc[:, 0].values if x is not None)
        combined_text = header_text + ' ' + col_text

        return any(keyword in combined_text for keyword in financial_keywords)

    def _process_financial_table(self, df: pd.DataFrame, table_name: str, financials: Dict[str, Any]) -> None:
        """Process a financial table.

        Args:
            df: DataFrame containing table data
            table_name: Name of the table
            financials: Dictionary to store financial data
        """
        try:
            # Try to identify the header row and column
            header_row = 0
            header_col = 0

            # Use the first row as headers
            df.columns = [str(x) if x is not None else f"col_{i}" for i, x in enumerate(df.iloc[header_row])]

            # Use the first column as index
            df = df.iloc[header_row+1:].set_index(df.columns[header_col])

            # Add table to financials
            if "tables" not in financials:
                financials["tables"] = {}

            financials["tables"][table_name] = df.to_dict(orient='index')

            # Try to extract metrics from the table
            self._extract_metrics_from_table(df, financials)
        except Exception as e:
            logger.warning(f"Error processing financial table: {e}")

    def _extract_metrics_from_table(self, df: pd.DataFrame, financials: Dict[str, Any]) -> None:
        """Extract metrics from a financial table.

        Args:
            df: DataFrame containing table data
            financials: Dictionary to store financial data
        """
        try:
            # Common financial metrics to look for in the index
            financial_metrics = {
                'revenue': ['revenue', 'sales', 'net revenue', 'total revenue'],
                'net_income': ['net income', 'net earnings', 'net profit', 'net loss'],
                'total_assets': ['total assets', 'assets total'],
                'total_liabilities': ['total liabilities', 'liabilities total'],
                'stockholders_equity': ['stockholders equity', 'shareholders equity', 'total equity']
            }

            # Look for metrics in the index
            for metric_name, keywords in financial_metrics.items():
                for idx in df.index:
                    idx_str = str(idx).lower()
                    if any(keyword in idx_str for keyword in keywords):
                        # Found a metric, extract the value from the first numeric column
                        for col in df.columns:
                            try:
                                value = float(df.loc[idx, col])

                                # Add to metrics
                                financials["metrics"][metric_name] = value

                                # Add to facts
                                fact = {
                                    "xbrl_tag": f"table_{metric_name}",
                                    "metric_name": metric_name,
                                    "value": value,
                                    "category": "extracted_from_table"
                                }
                                financials["facts"].append(fact)

                                # Found a value, move to next metric
                                break
                            except (ValueError, TypeError):
                                continue
        except Exception as e:
            logger.warning(f"Error extracting metrics from table: {e}")

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
                    standard_name = self._normalize_fact_name(concept)

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

    def _is_financial_fact(self, fact_id: str) -> bool:
        """Check if a fact is a financial fact.

        Args:
            fact_id: Fact ID

        Returns:
            True if it's a financial fact, False otherwise
        """
        # Check if it's a US GAAP fact
        if fact_id.startswith('us-gaap:'):
            return True

        # Check if it's an IFRS fact
        if fact_id.startswith('ifrs-full:'):
            return True

        # Check for other common financial fact prefixes
        common_prefixes = ['dei:', 'invest:', 'srt:']
        for prefix in common_prefixes:
            if fact_id.startswith(prefix):
                return True

        return False

    def _get_fact_value(self, fact: Any) -> Optional[float]:
        """Get the value of a fact.

        Args:
            fact: Fact object

        Returns:
            Fact value as float or None if not available
        """
        try:
            # Check if fact has value attribute
            if hasattr(fact, 'value'):
                value = fact.value

                # Convert to float if possible
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    # Remove commas and try to convert
                    value = value.replace(',', '')
                    return float(value)

            return None
        except (ValueError, TypeError):
            return None

    def _normalize_fact_name(self, fact_id: str) -> str:
        """Normalize fact name.

        Args:
            fact_id: Fact ID

        Returns:
            Normalized fact name
        """
        # Remove namespace prefix if present
        if ':' in fact_id:
            fact_id = fact_id.split(':')[1]

        # Convert camel case to snake case
        result = ''
        for i, char in enumerate(fact_id):
            if i > 0 and char.isupper() and fact_id[i-1].islower():
                result += '_'
            result += char.lower()

        return result

    def _get_fact_category(self, fact_id: str) -> str:
        """Get the category of a fact.

        Args:
            fact_id: Fact ID

        Returns:
            Category of the fact
        """
        # Common income statement facts
        income_statement_facts = [
            'Revenue', 'Sales', 'CostOfRevenue', 'GrossProfit', 'OperatingExpense',
            'OperatingIncome', 'NetIncome', 'EarningsPerShare'
        ]

        # Common balance sheet facts
        balance_sheet_facts = [
            'Assets', 'Liabilities', 'Equity', 'Cash', 'Inventory', 'AccountsReceivable',
            'AccountsPayable', 'LongTermDebt'
        ]

        # Common cash flow facts
        cash_flow_facts = [
            'CashFlow', 'OperatingCashFlow', 'InvestingCashFlow', 'FinancingCashFlow',
            'CapitalExpenditure'
        ]

        # Check category based on fact name
        fact_name = fact_id.split(':')[-1]

        for term in income_statement_facts:
            if term in fact_name:
                return 'income_statement'

        for term in balance_sheet_facts:
            if term in fact_name:
                return 'balance_sheet'

        for term in cash_flow_facts:
            if term in fact_name:
                return 'cash_flow'

        return 'other'

    def _extract_context_info(self, context: Any) -> Dict[str, Any]:
        """Extract context information.

        Args:
            context: Context object

        Returns:
            Dictionary with context information
        """
        result = {
            'period_type': None,
            'start_date': None,
            'end_date': None,
            'is_primary': False
        }

        try:
            # Extract period information
            if hasattr(context, 'period'):
                period = context.period

                if hasattr(period, 'instant'):
                    result['period_type'] = 'instant'
                    result['end_date'] = str(period.instant)
                elif hasattr(period, 'start_date') and hasattr(period, 'end_date'):
                    result['period_type'] = 'duration'
                    result['start_date'] = str(period.start_date)
                    result['end_date'] = str(period.end_date)

                    # Check if this is a primary reporting period
                    try:
                        from datetime import datetime
                        start = datetime.strptime(result['start_date'], '%Y-%m-%d')
                        end = datetime.strptime(result['end_date'], '%Y-%m-%d')
                        duration = (end - start).days

                        # Typically, quarterly reports are ~90 days, annual reports are ~365 days
                        if 80 <= duration <= 100 or 350 <= duration <= 380:
                            result['is_primary'] = True
                    except:
                        pass

            # Extract segment information
            if hasattr(context, 'segment'):
                segment = context.segment
                result['segment'] = str(segment)

                # If there's a segment, it's usually not the primary view
                result['is_primary'] = False

            # Extract context ID
            if hasattr(context, 'id'):
                result['context_id'] = context.id

        except Exception as e:
            logger.warning(f"Error extracting context info: {e}")

        return result

    def _is_primary_fact(self, fact_id: str, period_type: Optional[str]) -> bool:
        """Check if a fact is a primary fact.

        Args:
            fact_id: Fact ID
            period_type: Period type

        Returns:
            True if it's a primary fact, False otherwise
        """
        # Primary facts are typically duration facts for income statement
        # and instant facts for balance sheet
        if period_type == 'duration' and self._get_fact_category(fact_id) == 'income_statement':
            return True

        if period_type == 'instant' and self._get_fact_category(fact_id) == 'balance_sheet':
            return True

        # Check for common primary facts
        primary_facts = [
            'Revenue', 'NetIncome', 'Assets', 'Liabilities', 'Equity',
            'OperatingCashFlow', 'EarningsPerShare'
        ]

        fact_name = fact_id.split(':')[-1]
        for term in primary_facts:
            if term in fact_name:
                return True

        return False

    def _get_statement_category(self, statement_name: str) -> str:
        """Get the category of a statement.

        Args:
            statement_name: Name of the statement

        Returns:
            Category of the statement
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

        # Add more advanced ratios

        # Price to earnings ratio (if stock price available)
        if "net_income" in metrics and "weighted_average_shares_outstanding" in metrics:
            net_income = float(metrics["net_income"])
            shares = float(metrics["weighted_average_shares_outstanding"])
            if shares > 0 and net_income > 0:
                eps = net_income / shares
                ratios["earnings_per_share"] = eps

        # Debt to EBITDA
        if "total_liabilities" in metrics and "operating_income" in metrics and "depreciation_and_amortization" in metrics:
            debt = float(metrics["total_liabilities"])
            operating_income = float(metrics["operating_income"])
            depreciation = float(metrics["depreciation_and_amortization"])
            ebitda = operating_income + depreciation
            if ebitda > 0:
                ratios["debt_to_ebitda"] = debt / ebitda

        # Add ratios to financials
        financials["ratios"] = ratios
