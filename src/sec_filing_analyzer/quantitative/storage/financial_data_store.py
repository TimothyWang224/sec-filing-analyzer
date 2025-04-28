"""
Financial Data Store Module

This module provides functionality to store and query financial data
extracted from XBRL filings using DuckDB.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataStore:
    """
    Stores and queries financial data extracted from XBRL filings using DuckDB.
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the financial data store.

        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path or "data/financial_data.duckdb"
        self.conn = duckdb.connect(self.db_path)
        self._initialize_schema()
        logger.info(f"Initialized financial data store at {self.db_path}")

    def _initialize_schema(self) -> None:
        """Initialize the database schema."""
        try:
            # Read schema from SQL file
            schema_path = Path(__file__).parent / "financial_db_schema.sql"
            with open(schema_path, "r") as f:
                schema_sql = f.read()

            # Execute schema creation
            self.conn.execute(schema_sql)
            logger.info("Initialized financial database schema")
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            raise

    def store_company(
        self,
        ticker: str,
        name: Optional[str] = None,
        cik: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Store company information.

        Args:
            ticker: Company ticker symbol
            name: Company name
            cik: SEC CIK number
            **kwargs: Additional company attributes

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data
            data = {"ticker": ticker, "name": name or ticker, "cik": cik}

            # Add additional attributes
            for key, value in kwargs.items():
                if key in ["sector", "industry", "exchange", "sic"]:
                    data[key] = value

            # Insert or update company
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data.keys()])
            values = list(data.values())

            self.conn.execute(
                f"""
                INSERT OR REPLACE INTO companies ({columns})
                VALUES ({placeholders})
            """,
                values,
            )

            logger.info(f"Stored company information for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Error storing company {ticker}: {e}")
            return False

    def store_filing(self, filing_id: str, ticker: str, accession_number: str, **kwargs) -> bool:
        """Store filing information.

        Args:
            filing_id: Internal filing ID
            ticker: Company ticker symbol
            accession_number: SEC accession number
            **kwargs: Additional filing attributes

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data
            data = {
                "id": filing_id,
                "ticker": ticker,
                "accession_number": accession_number,
            }

            # Add additional attributes
            for key, value in kwargs.items():
                if key in [
                    "filing_type",
                    "filing_date",
                    "fiscal_year",
                    "fiscal_quarter",
                    "fiscal_period_end_date",
                    "document_url",
                    "has_xbrl",
                ]:
                    data[key] = value

            # Insert or update filing
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data.keys()])
            values = list(data.values())

            self.conn.execute(
                f"""
                INSERT OR REPLACE INTO filings ({columns})
                VALUES ({placeholders})
            """,
                values,
            )

            logger.info(f"Stored filing information for {ticker} {accession_number}")
            return True
        except Exception as e:
            logger.error(f"Error storing filing {filing_id}: {e}")
            return False

    def store_financial_facts(self, filing_id: str, facts: List[Dict[str, Any]]) -> int:
        """Store financial facts from XBRL data.

        Args:
            filing_id: Filing ID
            facts: List of financial facts

        Returns:
            Number of facts stored
        """
        try:
            count = 0
            for fact in facts:
                # Skip facts without values
                if "value" not in fact or fact["value"] is None:
                    continue

                # Prepare data
                data = {
                    "id": f"{filing_id}_{fact['xbrl_tag']}_{fact.get('context_id', 'default')}",
                    "filing_id": filing_id,
                    "xbrl_tag": fact["xbrl_tag"],
                    "metric_name": fact.get("metric_name", ""),
                    "value": fact["value"],
                    "unit": fact.get("unit", "USD"),
                    "period_type": fact.get("period_type", ""),
                    "start_date": fact.get("start_date"),
                    "end_date": fact.get("end_date"),
                    "segment": fact.get("segment", ""),
                    "context_id": fact.get("context_id", ""),
                }

                # Insert or update fact
                columns = ", ".join(data.keys())
                placeholders = ", ".join(["?" for _ in data.keys()])
                values = list(data.values())

                self.conn.execute(
                    f"""
                    INSERT OR REPLACE INTO financial_facts ({columns})
                    VALUES ({placeholders})
                """,
                    values,
                )
                count += 1

            logger.info(f"Stored {count} financial facts for filing {filing_id}")
            return count
        except Exception as e:
            logger.error(f"Error storing financial facts for filing {filing_id}: {e}")
            return 0

    def store_time_series_metrics(
        self,
        ticker: str,
        filing_id: str,
        fiscal_year: int,
        fiscal_quarter: int,
        metrics: Dict[str, Any],
    ) -> int:
        """Store time series metrics.

        Args:
            ticker: Company ticker symbol
            filing_id: Filing ID
            fiscal_year: Fiscal year
            fiscal_quarter: Fiscal quarter
            metrics: Dictionary of metrics

        Returns:
            Number of metrics stored
        """
        try:
            count = 0
            for metric_name, value in metrics.items():
                # Skip non-numeric values
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue

                # Insert or update metric
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO time_series_metrics
                    (ticker, metric_name, fiscal_year, fiscal_quarter, value, filing_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        ticker,
                        metric_name,
                        fiscal_year,
                        fiscal_quarter,
                        value,
                        filing_id,
                    ),
                )
                count += 1

            logger.info(f"Stored {count} time series metrics for {ticker} {fiscal_year}Q{fiscal_quarter}")
            return count
        except Exception as e:
            logger.error(f"Error storing time series metrics: {e}")
            return 0

    def store_financial_ratios(
        self,
        ticker: str,
        filing_id: str,
        fiscal_year: int,
        fiscal_quarter: int,
        ratios: Dict[str, float],
    ) -> int:
        """Store financial ratios.

        Args:
            ticker: Company ticker symbol
            filing_id: Filing ID
            fiscal_year: Fiscal year
            fiscal_quarter: Fiscal quarter
            ratios: Dictionary of ratios

        Returns:
            Number of ratios stored
        """
        try:
            count = 0
            for ratio_name, value in ratios.items():
                # Skip non-numeric values
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue

                # Insert or update ratio
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO financial_ratios
                    (ticker, ratio_name, fiscal_year, fiscal_quarter, value, filing_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (ticker, ratio_name, fiscal_year, fiscal_quarter, value, filing_id),
                )
                count += 1

            logger.info(f"Stored {count} financial ratios for {ticker} {fiscal_year}Q{fiscal_quarter}")
            return count
        except Exception as e:
            logger.error(f"Error storing financial ratios: {e}")
            return 0

    def store_xbrl_data(self, xbrl_data: Dict[str, Any]) -> bool:
        """Store all data extracted from XBRL.

        Args:
            xbrl_data: Dictionary of XBRL data

        Returns:
            True if successful, False otherwise
        """
        try:
            if "error" in xbrl_data:
                logger.warning(f"Skipping XBRL data with error: {xbrl_data['error']}")
                return False

            # Extract key information
            filing_id = xbrl_data["filing_id"]
            ticker = xbrl_data["ticker"]
            accession_number = xbrl_data["accession_number"]
            fiscal_year = xbrl_data.get("fiscal_year")
            fiscal_quarter = xbrl_data.get("fiscal_quarter")

            if not fiscal_year or not fiscal_quarter:
                logger.warning(f"Missing fiscal period for {ticker} {accession_number}")
                return False

            # Store company information
            self.store_company(ticker)

            # Store filing information
            self.store_filing(
                filing_id=filing_id,
                ticker=ticker,
                accession_number=accession_number,
                filing_type=xbrl_data.get("filing_type"),
                filing_date=xbrl_data.get("filing_date"),
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                document_url=xbrl_data.get("filing_url"),
                has_xbrl=True,
            )

            # Store financial facts
            facts = xbrl_data.get("facts", [])
            self.store_financial_facts(filing_id, facts)

            # Store time series metrics
            metrics = xbrl_data.get("metrics", {})
            self.store_time_series_metrics(
                ticker=ticker,
                filing_id=filing_id,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                metrics=metrics,
            )

            # Store financial ratios
            ratios = xbrl_data.get("ratios", {})
            self.store_financial_ratios(
                ticker=ticker,
                filing_id=filing_id,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                ratios=ratios,
            )

            logger.info(f"Successfully stored XBRL data for {ticker} {accession_number}")
            return True
        except Exception as e:
            logger.error(f"Error storing XBRL data: {e}")
            return False

    def get_company_metrics(
        self,
        ticker: str,
        metrics: Optional[List[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        quarterly: bool = True,
    ) -> pd.DataFrame:
        """Get time series metrics for a company.

        Args:
            ticker: Company ticker symbol
            metrics: List of metrics to retrieve (None for all)
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            quarterly: Whether to include quarterly data (False for annual only)

        Returns:
            DataFrame with time series metrics
        """
        try:
            # Build query
            query = """
                SELECT 
                    ticker, 
                    metric_name, 
                    fiscal_year, 
                    fiscal_quarter, 
                    value
                FROM time_series_metrics
                WHERE ticker = ?
            """
            params = [ticker]

            # Add metric filter
            if metrics:
                placeholders = ", ".join(["?" for _ in metrics])
                query += f" AND metric_name IN ({placeholders})"
                params.extend(metrics)

            # Add year filters
            if start_year:
                query += " AND fiscal_year >= ?"
                params.append(start_year)

            if end_year:
                query += " AND fiscal_year <= ?"
                params.append(end_year)

            # Add quarterly filter
            if not quarterly:
                query += " AND fiscal_quarter = 4"

            # Add ordering
            query += " ORDER BY fiscal_year, fiscal_quarter, metric_name"

            # Execute query
            result = self.conn.execute(query, params).fetchdf()

            # Pivot the result for easier analysis
            if not result.empty:
                result["period"] = result.apply(lambda x: f"{x['fiscal_year']}Q{x['fiscal_quarter']}", axis=1)
                pivoted = result.pivot(index="period", columns="metric_name", values="value").reset_index()

                # Add year and quarter columns
                pivoted["fiscal_year"] = pivoted["period"].apply(lambda x: int(x.split("Q")[0]))
                pivoted["fiscal_quarter"] = pivoted["period"].apply(lambda x: int(x.split("Q")[1]))

                # Sort by year and quarter
                pivoted = pivoted.sort_values(["fiscal_year", "fiscal_quarter"])

                return pivoted

            return result
        except Exception as e:
            logger.error(f"Error getting company metrics: {e}")
            return pd.DataFrame()

    def compare_companies(
        self,
        tickers: List[str],
        metric: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        quarterly: bool = False,
    ) -> pd.DataFrame:
        """Compare a metric across companies.

        Args:
            tickers: List of company tickers
            metric: Metric to compare
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            quarterly: Whether to include quarterly data (False for annual only)

        Returns:
            DataFrame with comparison data
        """
        try:
            # Build query
            placeholders = ", ".join(["?" for _ in tickers])
            query = f"""
                SELECT 
                    ticker, 
                    fiscal_year, 
                    fiscal_quarter, 
                    value
                FROM time_series_metrics
                WHERE ticker IN ({placeholders})
                AND metric_name = ?
            """
            params = tickers + [metric]

            # Add year filters
            if start_year:
                query += " AND fiscal_year >= ?"
                params.append(start_year)

            if end_year:
                query += " AND fiscal_year <= ?"
                params.append(end_year)

            # Add quarterly filter
            if not quarterly:
                query += " AND fiscal_quarter = 4"

            # Add ordering
            query += " ORDER BY fiscal_year, fiscal_quarter, ticker"

            # Execute query
            result = self.conn.execute(query, params).fetchdf()

            # Pivot the result for easier comparison
            if not result.empty:
                result["period"] = result.apply(lambda x: f"{x['fiscal_year']}Q{x['fiscal_quarter']}", axis=1)
                pivoted = result.pivot(index="period", columns="ticker", values="value").reset_index()

                # Add year and quarter columns
                pivoted["fiscal_year"] = pivoted["period"].apply(lambda x: int(x.split("Q")[0]))
                pivoted["fiscal_quarter"] = pivoted["period"].apply(lambda x: int(x.split("Q")[1]))

                # Sort by year and quarter
                pivoted = pivoted.sort_values(["fiscal_year", "fiscal_quarter"])

                return pivoted

            return result
        except Exception as e:
            logger.error(f"Error comparing companies: {e}")
            return pd.DataFrame()

    def get_financial_ratios(
        self,
        ticker: str,
        ratios: Optional[List[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        quarterly: bool = False,
    ) -> pd.DataFrame:
        """Get financial ratios for a company.

        Args:
            ticker: Company ticker symbol
            ratios: List of ratios to retrieve (None for all)
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            quarterly: Whether to include quarterly data (False for annual only)

        Returns:
            DataFrame with financial ratios
        """
        try:
            # Build query
            query = """
                SELECT 
                    ticker, 
                    ratio_name, 
                    fiscal_year, 
                    fiscal_quarter, 
                    value
                FROM financial_ratios
                WHERE ticker = ?
            """
            params = [ticker]

            # Add ratio filter
            if ratios:
                placeholders = ", ".join(["?" for _ in ratios])
                query += f" AND ratio_name IN ({placeholders})"
                params.extend(ratios)

            # Add year filters
            if start_year:
                query += " AND fiscal_year >= ?"
                params.append(start_year)

            if end_year:
                query += " AND fiscal_year <= ?"
                params.append(end_year)

            # Add quarterly filter
            if not quarterly:
                query += " AND fiscal_quarter = 4"

            # Add ordering
            query += " ORDER BY fiscal_year, fiscal_quarter, ratio_name"

            # Execute query
            result = self.conn.execute(query, params).fetchdf()

            # Pivot the result for easier analysis
            if not result.empty:
                result["period"] = result.apply(lambda x: f"{x['fiscal_year']}Q{x['fiscal_quarter']}", axis=1)
                pivoted = result.pivot(index="period", columns="ratio_name", values="value").reset_index()

                # Add year and quarter columns
                pivoted["fiscal_year"] = pivoted["period"].apply(lambda x: int(x.split("Q")[0]))
                pivoted["fiscal_quarter"] = pivoted["period"].apply(lambda x: int(x.split("Q")[1]))

                # Sort by year and quarter
                pivoted = pivoted.sort_values(["fiscal_year", "fiscal_quarter"])

                return pivoted

            return result
        except Exception as e:
            logger.error(f"Error getting financial ratios: {e}")
            return pd.DataFrame()

    def get_financial_facts(self, filing_id: str, tags: Optional[List[str]] = None) -> pd.DataFrame:
        """Get financial facts for a specific filing.

        Args:
            filing_id: Filing ID
            tags: List of XBRL tags to retrieve (None for all)

        Returns:
            DataFrame with financial facts
        """
        try:
            # Build query
            query = """
                SELECT 
                    xbrl_tag, 
                    metric_name, 
                    value, 
                    unit, 
                    period_type, 
                    start_date, 
                    end_date, 
                    segment
                FROM financial_facts
                WHERE filing_id = ?
            """
            params = [filing_id]

            # Add tag filter
            if tags:
                placeholders = ", ".join(["?" for _ in tags])
                query += f" AND xbrl_tag IN ({placeholders})"
                params.extend(tags)

            # Add ordering
            query += " ORDER BY xbrl_tag"

            # Execute query
            result = self.conn.execute(query, params).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error getting financial facts: {e}")
            return pd.DataFrame()

    def get_filing_info(
        self,
        ticker: Optional[str] = None,
        filing_type: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get filing information.

        Args:
            ticker: Company ticker symbol (None for all)
            filing_type: Filing type (None for all)
            start_year: Start year (inclusive)
            end_year: End year (inclusive)

        Returns:
            DataFrame with filing information
        """
        try:
            # Build query
            query = """
                SELECT 
                    f.id, 
                    f.ticker, 
                    c.name as company_name,
                    f.accession_number, 
                    f.filing_type, 
                    f.filing_date, 
                    f.fiscal_year, 
                    f.fiscal_quarter, 
                    f.document_url
                FROM filings f
                JOIN companies c ON f.ticker = c.ticker
                WHERE 1=1
            """
            params = []

            # Add filters
            if ticker:
                query += " AND f.ticker = ?"
                params.append(ticker)

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
            query += " ORDER BY f.filing_date DESC"

            # Execute query
            result = self.conn.execute(query, params).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Error getting filing info: {e}")
            return pd.DataFrame()

    def run_custom_query(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
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
            Dictionary with database statistics
        """
        try:
            stats = {}

            # Get company count
            stats["company_count"] = self.conn.execute("SELECT COUNT(*) as count FROM companies").fetchone()[0]

            # Get filing count
            stats["filing_count"] = self.conn.execute("SELECT COUNT(*) as count FROM filings").fetchone()[0]

            # Get fact count
            stats["fact_count"] = self.conn.execute("SELECT COUNT(*) as count FROM financial_facts").fetchone()[0]

            # Get time series count
            stats["time_series_count"] = self.conn.execute(
                "SELECT COUNT(*) as count FROM time_series_metrics"
            ).fetchone()[0]

            # Get ratio count
            stats["ratio_count"] = self.conn.execute("SELECT COUNT(*) as count FROM financial_ratios").fetchone()[0]

            # Get companies
            stats["companies"] = (
                self.conn.execute("SELECT ticker FROM companies ORDER BY ticker").fetchdf()["ticker"].tolist()
            )

            # Get filing types
            stats["filing_types"] = (
                self.conn.execute("SELECT DISTINCT filing_type FROM filings ORDER BY filing_type")
                .fetchdf()["filing_type"]
                .tolist()
            )

            # Get year range
            years = self.conn.execute(
                "SELECT MIN(fiscal_year) as min_year, MAX(fiscal_year) as max_year FROM filings"
            ).fetchone()
            stats["min_year"] = years[0]
            stats["max_year"] = years[1]

            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
