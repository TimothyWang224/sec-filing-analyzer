"""
Improved DuckDB Financial Store

A module for working with the improved DuckDB schema for financial data.
"""

import logging

# Add the project root to the Python path
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the DuckDB manager
from src.sec_filing_analyzer.utils.duckdb_manager import duckdb_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImprovedDuckDBStore:
    """
    An interface to store and query financial data using the improved DuckDB schema.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        batch_size: int = 100,
        read_only: bool = True,
    ):
        """Initialize the improved DuckDB financial store.

        Args:
            db_path: Path to the DuckDB database file
            batch_size: Size of batches for bulk operations
            read_only: Whether to open the database in read-only mode
        """
        self.db_path = db_path or "data/financial_data.duckdb"
        self.batch_size = batch_size
        self.read_only = read_only

        # Use the DuckDB manager to get a connection with the appropriate mode
        if read_only:
            self.conn = duckdb_manager.get_read_only_connection(self.db_path)
            logger.info(
                f"Initialized improved DuckDB financial store at {self.db_path} in read-only mode"
            )
        else:
            self.conn = duckdb_manager.get_read_write_connection(self.db_path)
            logger.info(
                f"Initialized improved DuckDB financial store at {self.db_path} in read-write mode"
            )

    def close(self):
        """Close the database connection."""
        # We don't actually close the connection since it's managed by the DuckDB manager
        # The DuckDB manager will handle connection pooling and closing when appropriate
        logger.info("Connection managed by DuckDB manager - not explicitly closing")

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()

    # Company-related methods

    def get_company_id(self, ticker: str) -> Optional[int]:
        """Get the company ID for a ticker.

        Args:
            ticker: Company ticker symbol

        Returns:
            Company ID or None if not found
        """
        try:
            result = self.conn.execute(
                "SELECT company_id FROM companies WHERE ticker = ?", [ticker]
            ).fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting company ID for {ticker}: {e}")
            return None

    def company_exists(self, ticker: str) -> bool:
        """Check if a company exists in the database.

        Args:
            ticker: Company ticker symbol

        Returns:
            True if the company exists, False otherwise
        """
        return self.get_company_id(ticker) is not None

    def store_company(self, company_data: Dict[str, Any]) -> Optional[int]:
        """Store company information.

        Args:
            company_data: Dictionary containing company data
                Required keys: ticker
                Optional keys: name, cik, sic, sector, industry, exchange

        Returns:
            Company ID if successful, None otherwise
        """
        try:
            ticker = company_data.get("ticker")
            if not ticker:
                logger.error("Ticker is required for storing company data")
                return None

            # Check if company already exists
            company_id = self.get_company_id(ticker)
            if company_id:
                # Update existing company
                self.conn.execute(
                    """
                    UPDATE companies SET
                        name = COALESCE(?, name),
                        cik = COALESCE(?, cik),
                        sic = COALESCE(?, sic),
                        sector = COALESCE(?, sector),
                        industry = COALESCE(?, industry),
                        exchange = COALESCE(?, exchange),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE company_id = ?
                    """,
                    [
                        company_data.get("name"),
                        company_data.get("cik"),
                        company_data.get("sic"),
                        company_data.get("sector"),
                        company_data.get("industry"),
                        company_data.get("exchange"),
                        company_id,
                    ],
                )
                logger.info(f"Updated company {ticker} (ID: {company_id})")
                return company_id
            else:
                # Insert new company
                # Get the next available company_id
                max_id = self.conn.execute(
                    "SELECT MAX(company_id) FROM companies"
                ).fetchone()[0]
                new_id = 1 if max_id is None else max_id + 1

                self.conn.execute(
                    """
                    INSERT INTO companies (
                        company_id, ticker, name, cik, sic, sector, industry, exchange,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    [
                        new_id,
                        ticker,
                        company_data.get("name"),
                        company_data.get("cik"),
                        company_data.get("sic"),
                        company_data.get("sector"),
                        company_data.get("industry"),
                        company_data.get("exchange"),
                    ],
                )
                logger.info(f"Inserted new company {ticker} (ID: {new_id})")
                return new_id
        except Exception as e:
            logger.error(f"Error storing company {company_data.get('ticker')}: {e}")
            return None

    def get_company(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get company information.

        Args:
            ticker: Company ticker symbol

        Returns:
            Dictionary containing company data or None if not found
        """
        try:
            result = self.conn.execute(
                """
                SELECT
                    company_id, ticker, name, cik, sic, sector, industry, exchange,
                    created_at, updated_at
                FROM companies
                WHERE ticker = ?
                """,
                [ticker],
            ).fetchone()

            if result:
                return {
                    "company_id": result[0],
                    "ticker": result[1],
                    "name": result[2],
                    "cik": result[3],
                    "sic": result[4],
                    "sector": result[5],
                    "industry": result[6],
                    "exchange": result[7],
                    "created_at": result[8],
                    "updated_at": result[9],
                }
            return None
        except Exception as e:
            logger.error(f"Error getting company {ticker}: {e}")
            return None

    def get_all_companies(self) -> pd.DataFrame:
        """Get all companies.

        Returns:
            DataFrame containing all companies
        """
        try:
            result = self.conn.execute(
                """
                SELECT
                    company_id, ticker, name, cik, sic, sector, industry, exchange
                FROM companies
                ORDER BY ticker
                """
            ).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error getting all companies: {e}")
            return pd.DataFrame()

    # Filing-related methods

    def get_filing_id(self, accession_number: str) -> Optional[int]:
        """Get the filing ID for an accession number.

        Args:
            accession_number: SEC accession number

        Returns:
            Filing ID or None if not found
        """
        try:
            result = self.conn.execute(
                "SELECT filing_id FROM filings WHERE accession_number = ?",
                [accession_number],
            ).fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting filing ID for {accession_number}: {e}")
            return None

    def filing_exists(self, accession_number: str) -> bool:
        """Check if a filing exists in the database.

        Args:
            accession_number: SEC accession number

        Returns:
            True if the filing exists, False otherwise
        """
        return self.get_filing_id(accession_number) is not None

    def store_filing(self, filing_data: Dict[str, Any]) -> Optional[int]:
        """Store filing information.

        Args:
            filing_data: Dictionary containing filing data
                Required keys: accession_number, company_id or ticker
                Optional keys: filing_type, filing_date, fiscal_year, fiscal_period,
                               fiscal_period_end_date, document_url, has_xbrl

        Returns:
            Filing ID if successful, None otherwise
        """
        try:
            accession_number = filing_data.get("accession_number")
            if not accession_number:
                logger.error("Accession number is required for storing filing data")
                return None

            # Get company_id from ticker if not provided
            company_id = filing_data.get("company_id")
            if not company_id and "ticker" in filing_data:
                company_id = self.get_company_id(filing_data["ticker"])
                if not company_id:
                    logger.error(
                        f"Company with ticker {filing_data['ticker']} not found"
                    )
                    return None

            if not company_id:
                logger.error(
                    "Either company_id or ticker is required for storing filing data"
                )
                return None

            # Check if filing already exists
            filing_id = self.get_filing_id(accession_number)
            if filing_id:
                # Update existing filing
                self.conn.execute(
                    """
                    UPDATE filings SET
                        company_id = COALESCE(?, company_id),
                        filing_type = COALESCE(?, filing_type),
                        filing_date = COALESCE(?, filing_date),
                        fiscal_year = COALESCE(?, fiscal_year),
                        fiscal_period = COALESCE(?, fiscal_period),
                        fiscal_period_end_date = COALESCE(?, fiscal_period_end_date),
                        document_url = COALESCE(?, document_url),
                        has_xbrl = COALESCE(?, has_xbrl),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE filing_id = ?
                    """,
                    [
                        company_id,
                        filing_data.get("filing_type"),
                        filing_data.get("filing_date"),
                        filing_data.get("fiscal_year"),
                        filing_data.get("fiscal_period"),
                        filing_data.get("fiscal_period_end_date"),
                        filing_data.get("document_url"),
                        filing_data.get("has_xbrl"),
                        filing_id,
                    ],
                )
                logger.info(f"Updated filing {accession_number} (ID: {filing_id})")
                return filing_id
            else:
                # Insert new filing
                # Get the next available filing_id
                max_id = self.conn.execute(
                    "SELECT MAX(filing_id) FROM filings"
                ).fetchone()[0]
                new_id = 1 if max_id is None else max_id + 1

                # Convert fiscal_quarter to fiscal_period if needed
                fiscal_period = filing_data.get("fiscal_period")
                if not fiscal_period and "fiscal_quarter" in filing_data:
                    fiscal_quarter = filing_data.get("fiscal_quarter")
                    fiscal_period = (
                        f"Q{fiscal_quarter}" if fiscal_quarter in [1, 2, 3] else "FY"
                    )

                self.conn.execute(
                    """
                    INSERT INTO filings (
                        filing_id, company_id, accession_number, filing_type, filing_date,
                        fiscal_year, fiscal_period, fiscal_period_end_date, document_url,
                        has_xbrl, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    [
                        new_id,
                        company_id,
                        accession_number,
                        filing_data.get("filing_type"),
                        filing_data.get("filing_date"),
                        filing_data.get("fiscal_year"),
                        fiscal_period,
                        filing_data.get("fiscal_period_end_date"),
                        filing_data.get("document_url"),
                        filing_data.get("has_xbrl", True),
                    ],
                )
                logger.info(f"Inserted new filing {accession_number} (ID: {new_id})")
                return new_id
        except Exception as e:
            logger.error(
                f"Error storing filing {filing_data.get('accession_number')}: {e}"
            )
            return None

    def get_filing(self, accession_number: str) -> Optional[Dict[str, Any]]:
        """Get filing information.

        Args:
            accession_number: SEC accession number

        Returns:
            Dictionary containing filing data or None if not found
        """
        try:
            result = self.conn.execute(
                """
                SELECT
                    f.filing_id, f.company_id, f.accession_number, f.filing_type, f.filing_date,
                    f.fiscal_year, f.fiscal_period, f.fiscal_period_end_date, f.document_url,
                    f.has_xbrl, f.created_at, f.updated_at, c.ticker
                FROM filings f
                JOIN companies c ON f.company_id = c.company_id
                WHERE f.accession_number = ?
                """,
                [accession_number],
            ).fetchone()

            if result:
                return {
                    "filing_id": result[0],
                    "company_id": result[1],
                    "accession_number": result[2],
                    "filing_type": result[3],
                    "filing_date": result[4],
                    "fiscal_year": result[5],
                    "fiscal_period": result[6],
                    "fiscal_period_end_date": result[7],
                    "document_url": result[8],
                    "has_xbrl": result[9],
                    "created_at": result[10],
                    "updated_at": result[11],
                    "ticker": result[12],
                }
            return None
        except Exception as e:
            logger.error(f"Error getting filing {accession_number}: {e}")
            return None

    def get_company_filings(
        self,
        ticker: str,
        filing_type: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get filings for a company.

        Args:
            ticker: Company ticker symbol
            filing_type: Filter by filing type (e.g., '10-K', '10-Q')
            start_year: Filter by fiscal year (inclusive)
            end_year: Filter by fiscal year (inclusive)

        Returns:
            DataFrame containing filings
        """
        try:
            # Build query
            query = """
                SELECT
                    f.filing_id, f.accession_number, f.filing_type, f.filing_date,
                    f.fiscal_year, f.fiscal_period, f.fiscal_period_end_date, f.document_url,
                    f.has_xbrl
                FROM filings f
                JOIN companies c ON f.company_id = c.company_id
                WHERE c.ticker = ?
            """
            params = [ticker]

            # Add filters
            if filing_type:
                query += " AND f.filing_type = ?"
                params.append(filing_type)

            if start_year:
                query += " AND f.fiscal_year >= ?"
                params.append(start_year)

            if end_year:
                query += " AND f.fiscal_year <= ?"
                params.append(end_year)

            # Add ordering
            query += " ORDER BY f.fiscal_year DESC, f.filing_date DESC"

            # Execute query
            result = self.conn.execute(query, params).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error getting filings for {ticker}: {e}")
            return pd.DataFrame()

    # Metrics-related methods

    def get_metric_id(self, metric_name: str) -> Optional[int]:
        """Get the metric ID for a metric name.

        Args:
            metric_name: Metric name

        Returns:
            Metric ID or None if not found
        """
        try:
            result = self.conn.execute(
                "SELECT metric_id FROM metrics WHERE metric_name = ?", [metric_name]
            ).fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting metric ID for {metric_name}: {e}")
            return None

    def metric_exists(self, metric_name: str) -> bool:
        """Check if a metric exists in the database.

        Args:
            metric_name: Metric name

        Returns:
            True if the metric exists, False otherwise
        """
        return self.get_metric_id(metric_name) is not None

    def store_metric(self, metric_data: Dict[str, Any]) -> Optional[int]:
        """Store metric information.

        Args:
            metric_data: Dictionary containing metric data
                Required keys: metric_name
                Optional keys: display_name, description, category, unit_of_measure,
                               is_calculated, calculation_formula

        Returns:
            Metric ID if successful, None otherwise
        """
        try:
            metric_name = metric_data.get("metric_name")
            if not metric_name:
                logger.error("Metric name is required for storing metric data")
                return None

            # Check if metric already exists
            metric_id = self.get_metric_id(metric_name)
            if metric_id:
                # Update existing metric
                self.conn.execute(
                    """
                    UPDATE metrics SET
                        display_name = COALESCE(?, display_name),
                        description = COALESCE(?, description),
                        category = COALESCE(?, category),
                        unit_of_measure = COALESCE(?, unit_of_measure),
                        is_calculated = COALESCE(?, is_calculated),
                        calculation_formula = COALESCE(?, calculation_formula),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE metric_id = ?
                    """,
                    [
                        metric_data.get("display_name"),
                        metric_data.get("description"),
                        metric_data.get("category"),
                        metric_data.get("unit_of_measure"),
                        metric_data.get("is_calculated"),
                        metric_data.get("calculation_formula"),
                        metric_id,
                    ],
                )
                logger.info(f"Updated metric {metric_name} (ID: {metric_id})")
                return metric_id
            else:
                # Insert new metric
                # Get the next available metric_id
                max_id = self.conn.execute(
                    "SELECT MAX(metric_id) FROM metrics"
                ).fetchone()[0]
                new_id = 1 if max_id is None else max_id + 1

                # Generate display name if not provided
                display_name = metric_data.get("display_name")
                if not display_name:
                    display_name = " ".join(
                        word.capitalize() for word in metric_name.split("_")
                    )

                self.conn.execute(
                    """
                    INSERT INTO metrics (
                        metric_id, metric_name, display_name, description, category,
                        unit_of_measure, is_calculated, calculation_formula,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    [
                        new_id,
                        metric_name,
                        display_name,
                        metric_data.get("description", display_name),
                        metric_data.get("category", "other"),
                        metric_data.get("unit_of_measure", "USD"),
                        metric_data.get("is_calculated", False),
                        metric_data.get("calculation_formula"),
                    ],
                )
                logger.info(f"Inserted new metric {metric_name} (ID: {new_id})")
                return new_id
        except Exception as e:
            logger.error(f"Error storing metric {metric_data.get('metric_name')}: {e}")
            return None

    def get_metric(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get metric information.

        Args:
            metric_name: Metric name

        Returns:
            Dictionary containing metric data or None if not found
        """
        try:
            result = self.conn.execute(
                """
                SELECT
                    metric_id, metric_name, display_name, description, category,
                    unit_of_measure, is_calculated, calculation_formula,
                    created_at, updated_at
                FROM metrics
                WHERE metric_name = ?
                """,
                [metric_name],
            ).fetchone()

            if result:
                return {
                    "metric_id": result[0],
                    "metric_name": result[1],
                    "display_name": result[2],
                    "description": result[3],
                    "category": result[4],
                    "unit_of_measure": result[5],
                    "is_calculated": result[6],
                    "calculation_formula": result[7],
                    "created_at": result[8],
                    "updated_at": result[9],
                }
            return None
        except Exception as e:
            logger.error(f"Error getting metric {metric_name}: {e}")
            return None

    def get_all_metrics(self, category: Optional[str] = None) -> pd.DataFrame:
        """Get all metrics.

        Args:
            category: Filter by category (e.g., 'income_statement', 'balance_sheet')

        Returns:
            DataFrame containing metrics
        """
        try:
            # Build query
            query = """
                SELECT
                    metric_id, metric_name, display_name, description, category,
                    unit_of_measure, is_calculated
                FROM metrics
            """
            params = []

            # Add category filter
            if category:
                query += " WHERE category = ?"
                params.append(category)

            # Add ordering
            query += " ORDER BY category, metric_name"

            # Execute query
            result = self.conn.execute(query, params).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error getting all metrics: {e}")
            return pd.DataFrame()

    # Facts-related methods

    def store_fact(self, fact_data: Dict[str, Any]) -> Optional[int]:
        """Store a financial fact.

        Args:
            fact_data: Dictionary containing fact data
                Required keys: filing_id, metric_id or metric_name, value
                Optional keys: as_reported, normalized_value, period_type, start_date,
                               end_date, context_id, decimals

        Returns:
            Fact ID if successful, None otherwise
        """
        try:
            # Handle special values like 'INF', '-INF', and 'NaN'
            # Convert special values in 'value' field
            if "value" in fact_data:
                if isinstance(fact_data["value"], str):
                    upper_value = fact_data["value"].upper()
                    if upper_value == "INF":
                        fact_data["value"] = float("inf")
                        logger.debug(
                            f"Converted 'INF' value to float infinity for fact {fact_data.get('metric_name')}"
                        )
                    elif upper_value == "-INF":
                        fact_data["value"] = float("-inf")
                        logger.debug(
                            f"Converted '-INF' value to negative float infinity for fact {fact_data.get('metric_name')}"
                        )
                    elif upper_value in ("NAN", "NA", "N/A"):
                        fact_data["value"] = float("nan")
                        logger.debug(
                            f"Converted '{upper_value}' value to NaN for fact {fact_data.get('metric_name')}"
                        )

            # Convert special values in 'decimals' field
            if "decimals" in fact_data:
                if isinstance(fact_data["decimals"], str):
                    upper_decimals = fact_data["decimals"].upper()
                    if upper_decimals in ("INF", "-INF", "NAN", "NA", "N/A"):
                        fact_data["decimals"] = None  # Use NULL for special values
                        logger.debug(
                            f"Converted '{fact_data['decimals']}' decimals to NULL for fact {fact_data.get('metric_name')}"
                        )

            # Handle special values in 'normalized_value' field
            if "normalized_value" in fact_data:
                if isinstance(fact_data["normalized_value"], str):
                    upper_norm_value = fact_data["normalized_value"].upper()
                    if upper_norm_value == "INF":
                        fact_data["normalized_value"] = float("inf")
                        logger.debug(
                            f"Converted 'INF' normalized_value to float infinity for fact {fact_data.get('metric_name')}"
                        )
                    elif upper_norm_value == "-INF":
                        fact_data["normalized_value"] = float("-inf")
                        logger.debug(
                            f"Converted '-INF' normalized_value to negative float infinity for fact {fact_data.get('metric_name')}"
                        )
                    elif upper_norm_value in ("NAN", "NA", "N/A"):
                        fact_data["normalized_value"] = float("nan")
                        logger.debug(
                            f"Converted '{upper_norm_value}' normalized_value to NaN for fact {fact_data.get('metric_name')}"
                        )

            # Check required fields
            filing_id = fact_data.get("filing_id")
            if not filing_id:
                logger.error("Filing ID is required for storing fact data")
                return None

            # Get metric_id from metric_name if not provided
            metric_id = fact_data.get("metric_id")
            if not metric_id and "metric_name" in fact_data:
                metric_id = self.get_metric_id(fact_data["metric_name"])
                if not metric_id:
                    # Create the metric if it doesn't exist
                    metric_id = self.store_metric(
                        {"metric_name": fact_data["metric_name"]}
                    )
                    if not metric_id:
                        logger.error(
                            f"Failed to create metric {fact_data['metric_name']}"
                        )
                        return None

            if not metric_id:
                logger.error(
                    "Either metric_id or metric_name is required for storing fact data"
                )
                return None

            value = fact_data.get("value")
            if value is None:
                logger.error("Value is required for storing fact data")
                return None

            # Check if fact already exists
            context_id = fact_data.get("context_id", "")
            existing_fact = self.conn.execute(
                """
                SELECT fact_id FROM facts
                WHERE filing_id = ? AND metric_id = ? AND context_id = ?
                """,
                [filing_id, metric_id, context_id],
            ).fetchone()

            if existing_fact:
                # Update existing fact
                fact_id = existing_fact[0]
                self.conn.execute(
                    """
                    UPDATE facts SET
                        value = ?,
                        as_reported = COALESCE(?, as_reported),
                        normalized_value = COALESCE(?, normalized_value),
                        period_type = COALESCE(?, period_type),
                        start_date = COALESCE(?, start_date),
                        end_date = COALESCE(?, end_date),
                        decimals = COALESCE(?, decimals),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE fact_id = ?
                    """,
                    [
                        value,
                        fact_data.get("as_reported"),
                        fact_data.get("normalized_value", value),
                        fact_data.get("period_type"),
                        fact_data.get("start_date"),
                        fact_data.get("end_date"),
                        fact_data.get("decimals"),
                        fact_id,
                    ],
                )
                logger.debug(f"Updated fact ID: {fact_id}")
                return fact_id
            else:
                # Insert new fact
                # Get the next available fact_id
                max_id = self.conn.execute("SELECT MAX(fact_id) FROM facts").fetchone()[
                    0
                ]
                new_id = 1 if max_id is None else max_id + 1

                self.conn.execute(
                    """
                    INSERT INTO facts (
                        fact_id, filing_id, metric_id, value, as_reported,
                        normalized_value, period_type, start_date, end_date,
                        context_id, decimals, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    [
                        new_id,
                        filing_id,
                        metric_id,
                        value,
                        fact_data.get("as_reported", True),
                        fact_data.get("normalized_value", value),
                        fact_data.get("period_type"),
                        fact_data.get("start_date"),
                        fact_data.get("end_date"),
                        context_id,
                        fact_data.get("decimals"),
                    ],
                )
                logger.debug(f"Inserted new fact ID: {new_id}")
                return new_id
        except Exception as e:
            logger.error(f"Error storing fact: {e}")
            return None

    def store_facts_batch(self, facts: List[Dict[str, Any]]) -> int:
        """Store multiple financial facts in a batch.

        Args:
            facts: List of dictionaries containing fact data

        Returns:
            Number of facts successfully stored
        """
        if not facts:
            return 0

        successful = 0
        for i in range(0, len(facts), self.batch_size):
            batch = facts[i : i + self.batch_size]
            for fact in batch:
                if self.store_fact(fact) is not None:
                    successful += 1

        logger.info(f"Stored {successful} out of {len(facts)} facts")
        return successful

    def get_filing_facts(
        self, filing_id: int, metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Get facts for a filing.

        Args:
            filing_id: Filing ID
            metric_names: List of metric names to filter by

        Returns:
            DataFrame containing facts
        """
        try:
            # Build query
            query = """
                SELECT
                    f.fact_id, f.filing_id, f.metric_id, f.value, f.as_reported,
                    f.normalized_value, f.period_type, f.start_date, f.end_date,
                    f.context_id, f.decimals, m.metric_name, m.display_name,
                    m.category, m.unit_of_measure
                FROM facts f
                JOIN metrics m ON f.metric_id = m.metric_id
                WHERE f.filing_id = ?
            """
            params = [filing_id]

            # Add metric filter
            if metric_names:
                placeholders = ", ".join(["?" for _ in metric_names])
                query += f" AND m.metric_name IN ({placeholders})"
                params.extend(metric_names)

            # Add ordering
            query += " ORDER BY m.category, m.metric_name"

            # Execute query
            result = self.conn.execute(query, params).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error getting facts for filing {filing_id}: {e}")
            return pd.DataFrame()

    # Query methods

    def query_time_series(
        self,
        ticker: str,
        metric_names: Optional[List[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        include_quarterly: bool = False,
    ) -> pd.DataFrame:
        """Query time series data for a company.

        Args:
            ticker: Company ticker symbol
            metric_names: List of metric names to filter by
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            include_quarterly: Whether to include quarterly data

        Returns:
            DataFrame containing time series data
        """
        try:
            # Build query using the time_series_view
            query = """
                SELECT
                    ticker, fiscal_year, fiscal_period, metric_name, display_name,
                    category, value, unit_of_measure, end_date
                FROM time_series_view
                WHERE ticker = ?
            """
            params = [ticker]

            # Add metric filter
            if metric_names:
                placeholders = ", ".join(["?" for _ in metric_names])
                query += f" AND metric_name IN ({placeholders})"
                params.extend(metric_names)

            # Add year filters
            if start_year:
                query += " AND fiscal_year >= ?"
                params.append(start_year)

            if end_year:
                query += " AND fiscal_year <= ?"
                params.append(end_year)

            # Add quarterly filter
            if not include_quarterly:
                query += " AND fiscal_period = 'FY'"

            # Add ordering
            query += " ORDER BY fiscal_year, fiscal_period, category, metric_name"

            # Execute query
            result = self.conn.execute(query, params).fetchdf()

            # Pivot the result for easier analysis
            if not result.empty:
                result["period"] = result.apply(
                    lambda x: f"{x['fiscal_year']}-{x['fiscal_period']}", axis=1
                )
                pivoted = result.pivot(
                    index="period", columns="metric_name", values="value"
                ).reset_index()

                # Add year and period columns
                pivoted["fiscal_year"] = pivoted["period"].apply(
                    lambda x: int(x.split("-")[0])
                )
                pivoted["fiscal_period"] = pivoted["period"].apply(
                    lambda x: x.split("-")[1]
                )

                # Sort by year and period
                pivoted = pivoted.sort_values(["fiscal_year", "fiscal_period"])

                return pivoted

            return result
        except Exception as e:
            logger.error(f"Error querying time series for {ticker}: {e}")
            return pd.DataFrame()

    def query_company_comparison(
        self,
        tickers: List[str],
        metric_name: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        include_quarterly: bool = False,
    ) -> pd.DataFrame:
        """Query data for comparing companies.

        Args:
            tickers: List of company ticker symbols
            metric_name: Metric name to compare
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            include_quarterly: Whether to include quarterly data

        Returns:
            DataFrame containing comparison data
        """
        try:
            # Build query using the company_comparison_view
            query = """
                SELECT
                    fiscal_year, fiscal_period, ticker, value, unit_of_measure
                FROM company_comparison_view
                WHERE metric_name = ?
                AND ticker IN ({placeholders})
            """.format(placeholders=", ".join(["?" for _ in tickers]))

            params = [metric_name] + tickers

            # Add year filters
            if start_year:
                query += " AND fiscal_year >= ?"
                params.append(start_year)

            if end_year:
                query += " AND fiscal_year <= ?"
                params.append(end_year)

            # Add quarterly filter
            if not include_quarterly:
                query += " AND fiscal_period = 'FY'"

            # Add ordering
            query += " ORDER BY fiscal_year, fiscal_period, ticker"

            # Execute query
            result = self.conn.execute(query, params).fetchdf()

            # Pivot the result for easier comparison
            if not result.empty:
                result["period"] = result.apply(
                    lambda x: f"{x['fiscal_year']}-{x['fiscal_period']}", axis=1
                )
                pivoted = result.pivot(
                    index="period", columns="ticker", values="value"
                ).reset_index()

                # Add year and period columns
                pivoted["fiscal_year"] = pivoted["period"].apply(
                    lambda x: int(x.split("-")[0])
                )
                pivoted["fiscal_period"] = pivoted["period"].apply(
                    lambda x: x.split("-")[1]
                )

                # Sort by year and period
                pivoted = pivoted.sort_values(["fiscal_year", "fiscal_period"])

                return pivoted

            return result
        except Exception as e:
            logger.error(f"Error querying company comparison: {e}")
            return pd.DataFrame()

    def query_latest_metrics(
        self, ticker: str, category: Optional[str] = None
    ) -> pd.DataFrame:
        """Query the latest metrics for a company.

        Args:
            ticker: Company ticker symbol
            category: Filter by category (e.g., 'income_statement', 'balance_sheet')

        Returns:
            DataFrame containing the latest metrics
        """
        try:
            # Find the latest filing
            latest_filing_query = """
                SELECT MAX(filing_id) AS filing_id
                FROM filings f
                JOIN companies c ON f.company_id = c.company_id
                WHERE c.ticker = ?
                AND f.fiscal_period = 'FY'
            """
            latest_filing = self.conn.execute(latest_filing_query, [ticker]).fetchone()

            if not latest_filing or not latest_filing[0]:
                logger.warning(f"No filings found for {ticker}")
                return pd.DataFrame()

            filing_id = latest_filing[0]

            # Build query
            query = """
                SELECT
                    m.category, m.display_name, f.value, m.unit_of_measure,
                    fi.fiscal_year, fi.fiscal_period
                FROM facts f
                JOIN metrics m ON f.metric_id = m.metric_id
                JOIN filings fi ON f.filing_id = fi.filing_id
                WHERE f.filing_id = ?
            """
            params = [filing_id]

            # Add category filter
            if category:
                query += " AND m.category = ?"
                params.append(category)

            # Add ordering
            query += " ORDER BY m.category, m.display_name"

            # Execute query
            result = self.conn.execute(query, params).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error querying latest metrics for {ticker}: {e}")
            return pd.DataFrame()

    def run_custom_query(
        self, query: str, params: Optional[List[Any]] = None
    ) -> pd.DataFrame:
        """Run a custom SQL query.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            DataFrame with query results
        """
        try:
            result = self.conn.execute(query, params or []).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error running custom query: {e}")
            return pd.DataFrame()

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary containing database statistics
        """
        try:
            stats = {}

            # Count companies
            companies_count = self.conn.execute(
                "SELECT COUNT(*) FROM companies"
            ).fetchone()[0]
            stats["companies_count"] = companies_count

            # Count filings
            filings_count = self.conn.execute(
                "SELECT COUNT(*) FROM filings"
            ).fetchone()[0]
            stats["filings_count"] = filings_count

            # Count metrics
            metrics_count = self.conn.execute(
                "SELECT COUNT(*) FROM metrics"
            ).fetchone()[0]
            stats["metrics_count"] = metrics_count

            # Count facts
            facts_count = self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            stats["facts_count"] = facts_count

            # Get year range
            year_range = self.conn.execute(
                "SELECT MIN(fiscal_year), MAX(fiscal_year) FROM filings"
            ).fetchone()
            stats["min_year"] = year_range[0]
            stats["max_year"] = year_range[1]

            # Get filing types
            filing_types = self.conn.execute(
                "SELECT DISTINCT filing_type FROM filings ORDER BY filing_type"
            ).fetchall()
            stats["filing_types"] = [ft[0] for ft in filing_types]

            # Get metric categories
            metric_categories = self.conn.execute(
                "SELECT DISTINCT category FROM metrics ORDER BY category"
            ).fetchall()
            stats["metric_categories"] = [mc[0] for mc in metric_categories]

            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
