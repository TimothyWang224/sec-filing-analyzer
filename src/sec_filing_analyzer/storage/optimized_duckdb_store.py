"""
Optimized DuckDB Financial Store Module

This module provides an optimized interface to store and query financial data
extracted from SEC filings using DuckDB with batch operations.
"""

import logging

# Add the project root to the Python path
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the DuckDB manager
from sec_filing_analyzer.utils.duckdb_manager import duckdb_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedDuckDBStore:
    """
    An optimized interface to store and query financial data using DuckDB with batch operations.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        batch_size: int = 100,
        read_only: bool = True,
    ):
        """Initialize the optimized DuckDB financial store.

        Args:
            db_path: Path to the DuckDB database file
            batch_size: Size of batches for bulk operations
            read_only: Whether to open the database in read-only mode
        """
        self.db_path = db_path or "data/db_backup/financial_data.duckdb"
        self.batch_size = batch_size
        self.read_only = read_only

        # Use the DuckDB manager to get a connection with the appropriate mode
        if read_only:
            self.conn = duckdb_manager.get_read_only_connection(self.db_path)
            logger.info(f"Initialized optimized DuckDB financial store at {self.db_path} in read-only mode")
        else:
            self.conn = duckdb_manager.get_read_write_connection(self.db_path)
            # Initialize schema only in read-write mode
            self._initialize_schema()
            logger.info(f"Initialized optimized DuckDB financial store at {self.db_path} in read-write mode")

    def _initialize_schema(self):
        """Initialize the database schema."""
        try:
            # Create companies table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    ticker VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    cik VARCHAR,
                    sic VARCHAR,
                    sector VARCHAR,
                    industry VARCHAR,
                    exchange VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create filings table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS filings (
                    id VARCHAR PRIMARY KEY,
                    ticker VARCHAR,
                    accession_number VARCHAR,
                    filing_type VARCHAR,
                    filing_date DATE,
                    fiscal_year INTEGER,
                    fiscal_quarter INTEGER,
                    fiscal_period_end_date DATE,
                    document_url VARCHAR,
                    has_xbrl BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (ticker) REFERENCES companies(ticker)
                )
            """)

            # Create financial facts table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS financial_facts (
                    id VARCHAR PRIMARY KEY,
                    filing_id VARCHAR,
                    xbrl_tag VARCHAR,
                    metric_name VARCHAR,
                    value DOUBLE,
                    unit VARCHAR,
                    period_type VARCHAR,
                    start_date DATE,
                    end_date DATE,
                    segment VARCHAR,
                    context_id VARCHAR,
                    decimals INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (filing_id) REFERENCES filings(id)
                )
            """)

            # Create time series metrics table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS time_series_metrics (
                    ticker VARCHAR,
                    metric_name VARCHAR,
                    fiscal_year INTEGER,
                    fiscal_quarter INTEGER,
                    value DOUBLE,
                    unit VARCHAR,
                    filing_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, metric_name, fiscal_year, fiscal_quarter),
                    FOREIGN KEY (ticker) REFERENCES companies(ticker),
                    FOREIGN KEY (filing_id) REFERENCES filings(id)
                )
            """)

            # Create financial ratios table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS financial_ratios (
                    ticker VARCHAR,
                    fiscal_year INTEGER,
                    fiscal_quarter INTEGER,
                    ratio_name VARCHAR,
                    value DOUBLE,
                    filing_id VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (ticker, ratio_name, fiscal_year, fiscal_quarter),
                    FOREIGN KEY (ticker) REFERENCES companies(ticker),
                    FOREIGN KEY (filing_id) REFERENCES filings(id)
                )
            """)

            # Create XBRL tag mappings table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS xbrl_tag_mappings (
                    xbrl_tag VARCHAR PRIMARY KEY,
                    standard_metric_name VARCHAR,
                    category VARCHAR,
                    description VARCHAR,
                    is_custom BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            logger.info("Initialized DuckDB schema")
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            raise

    def store_company(self, company_data: Dict[str, Any]) -> bool:
        """
        Store a single company in the database.

        Args:
            company_data: Dictionary with company data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Store the company using the batch method
            result = self.store_companies_batch([company_data])
            return result > 0
        except Exception as e:
            logger.error(f"Error storing company: {e}")
            return False

    def store_companies_batch(self, companies: List[Dict[str, Any]]) -> int:
        """Store multiple companies in a batch.

        Args:
            companies: List of company dictionaries

        Returns:
            Number of companies stored
        """
        try:
            if not companies:
                return 0

            # Prepare data for batch insert
            data = []
            for company in companies:
                ticker = company.get("ticker")
                if not ticker:
                    continue

                row = {
                    "ticker": ticker,
                    "name": company.get("name", ticker),
                    "cik": company.get("cik"),
                    "sic": company.get("sic"),
                    "sector": company.get("sector"),
                    "industry": company.get("industry"),
                    "exchange": company.get("exchange"),
                }
                data.append(row)

            if not data:
                return 0

            # Convert to DataFrame for batch insert
            df = pd.DataFrame(data)

            # Insert or replace companies
            # Get the column names from the companies table
            columns_result = self.conn.execute("PRAGMA table_info(companies)").fetchdf()
            column_names = columns_result["name"].tolist()

            # Remove the created_at column as it has a default value
            if "created_at" in column_names:
                column_names.remove("created_at")

            # Create a temporary table with the same structure
            self.conn.execute(f"""
                CREATE TEMPORARY TABLE IF NOT EXISTS temp_companies AS
                SELECT {", ".join(column_names)} FROM companies LIMIT 0
            """)

            # Register the DataFrame and insert into temp table
            self.conn.register("temp_df", df)

            # Only insert columns that exist in the DataFrame
            df_columns = df.columns.tolist()
            common_columns = [col for col in column_names if col in df_columns]

            self.conn.execute(f"""
                INSERT INTO temp_companies ({", ".join(common_columns)})
                SELECT {", ".join(common_columns)} FROM temp_df
            """)

            # Insert or replace into the main table
            self.conn.execute(f"""
                INSERT OR REPLACE INTO companies ({", ".join(common_columns)})
                SELECT {", ".join(common_columns)} FROM temp_companies
            """)

            # Drop the temporary table
            self.conn.execute("DROP TABLE IF EXISTS temp_companies")

            logger.info(f"Stored {len(data)} companies in batch")
            return len(data)
        except Exception as e:
            logger.error(f"Error storing companies batch: {e}")
            return 0

    def store_filing(self, filing_data: Dict[str, Any]) -> bool:
        """
        Store a single filing in the database.

        Args:
            filing_data: Dictionary with filing data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Store the filing using the batch method
            result = self.store_filings_batch([filing_data])
            return result > 0
        except Exception as e:
            logger.error(f"Error storing filing: {e}")
            return False

    def update_filing(self, filing_data: Dict[str, Any]) -> bool:
        """
        Update a filing in the database.

        Args:
            filing_data: Dictionary with filing data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the filing ID
            filing_id = filing_data.get("id")
            if not filing_id:
                logger.error("Filing ID is required for update")
                return False

            # Build the SET clause
            set_clause = ", ".join([f"{key} = ?" for key in filing_data.keys() if key != "id"])

            # Build the parameter list
            params = [filing_data[key] for key in filing_data.keys() if key != "id"]
            params.append(filing_id)  # Add the ID for the WHERE clause

            # Execute the update
            self.conn.execute(f"UPDATE filings SET {set_clause} WHERE id = ?", params)

            return True
        except Exception as e:
            logger.error(f"Error updating filing: {e}")
            return False

    def store_filings_batch(self, filings: List[Dict[str, Any]]) -> int:
        """Store multiple filings in a batch.

        Args:
            filings: List of filing dictionaries

        Returns:
            Number of filings stored
        """
        try:
            if not filings:
                return 0

            # Ensure companies exist
            companies = []
            for filing in filings:
                ticker = filing.get("ticker")
                if ticker:
                    companies.append({"ticker": ticker})

            self.store_companies_batch(companies)

            # Prepare data for batch insert
            data = []
            for filing in filings:
                filing_id = filing.get("id")
                ticker = filing.get("ticker")
                accession_number = filing.get("accession_number")

                if not filing_id or not ticker or not accession_number:
                    continue

                row = {
                    "id": filing_id,
                    "ticker": ticker,
                    "accession_number": accession_number,
                    "filing_type": filing.get("filing_type"),
                    "filing_date": filing.get("filing_date"),
                    "fiscal_year": filing.get("fiscal_year"),
                    "fiscal_quarter": filing.get("fiscal_quarter"),
                    "fiscal_period_end_date": filing.get("fiscal_period_end_date"),
                    "document_url": filing.get("document_url"),
                    "has_xbrl": filing.get("has_xbrl", True),
                }
                data.append(row)

            if not data:
                return 0

            # Convert to DataFrame for batch insert
            df = pd.DataFrame(data)

            # Insert or replace filings
            # Get the column names from the filings table
            columns_result = self.conn.execute("PRAGMA table_info(filings)").fetchdf()
            column_names = columns_result["name"].tolist()

            # Remove the created_at column as it has a default value
            if "created_at" in column_names:
                column_names.remove("created_at")

            # Create a temporary table with the same structure
            self.conn.execute(f"""
                CREATE TEMPORARY TABLE IF NOT EXISTS temp_filings AS
                SELECT {", ".join(column_names)} FROM filings LIMIT 0
            """)

            # Register the DataFrame and insert into temp table
            self.conn.register("temp_df", df)

            # Only insert columns that exist in the DataFrame
            df_columns = df.columns.tolist()
            common_columns = [col for col in column_names if col in df_columns]

            self.conn.execute(f"""
                INSERT INTO temp_filings ({", ".join(common_columns)})
                SELECT {", ".join(common_columns)} FROM temp_df
            """)

            # Insert or replace into the main table
            self.conn.execute(f"""
                INSERT OR REPLACE INTO filings ({", ".join(common_columns)})
                SELECT {", ".join(common_columns)} FROM temp_filings
            """)

            # Drop the temporary table
            self.conn.execute("DROP TABLE IF EXISTS temp_filings")

            logger.info(f"Stored {len(data)} filings in batch")
            return len(data)
        except Exception as e:
            logger.error(f"Error storing filings batch: {e}")
            return 0

    def store_financial_fact(self, fact_data: Dict[str, Any]) -> bool:
        """
        Store a single financial fact in the database.

        Args:
            fact_data: Dictionary with financial fact data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Store the fact using the batch method
            result = self.store_financial_facts_batch([fact_data])
            return result > 0
        except Exception as e:
            logger.error(f"Error storing financial fact: {e}")
            return False

    def store_financial_facts_batch(self, facts: List[Dict[str, Any]]) -> int:
        """Store multiple financial facts in a batch.

        Args:
            facts: List of fact dictionaries with filing_id, xbrl_tag, etc.

        Returns:
            Number of facts stored
        """
        try:
            if not facts:
                return 0

            # Prepare data for batch insert
            data = []
            for fact in facts:
                filing_id = fact.get("filing_id")
                xbrl_tag = fact.get("xbrl_tag")

                if not filing_id or not xbrl_tag:
                    continue

                # Skip facts without values
                value = fact.get("value")
                if value is None:
                    continue

                # Generate a unique ID if not provided
                fact_id = fact.get("id") or f"{filing_id}_{xbrl_tag}_{fact.get('context_id', 'default')}"

                row = {
                    "id": fact_id,
                    "filing_id": filing_id,
                    "xbrl_tag": xbrl_tag,
                    "metric_name": fact.get("metric_name", ""),
                    "value": value,
                    "unit": fact.get("unit", "USD"),
                    "period_type": fact.get("period_type", ""),
                    "start_date": fact.get("start_date"),
                    "end_date": fact.get("end_date"),
                    "segment": fact.get("segment", ""),
                    "context_id": fact.get("context_id", ""),
                }
                data.append(row)

            if not data:
                return 0

            # Process in batches to avoid memory issues
            total_stored = 0
            for i in range(0, len(data), self.batch_size):
                batch = data[i : i + self.batch_size]

                # Convert to DataFrame for batch insert
                df = pd.DataFrame(batch)

                # Insert or replace facts
                # Get the column names from the financial_facts table
                columns_result = self.conn.execute("PRAGMA table_info(financial_facts)").fetchdf()
                column_names = columns_result["name"].tolist()

                # Remove the created_at column as it has a default value
                if "created_at" in column_names:
                    column_names.remove("created_at")

                # Create a temporary table with the same structure
                self.conn.execute(f"""
                    CREATE TEMPORARY TABLE IF NOT EXISTS temp_facts AS
                    SELECT {", ".join(column_names)} FROM financial_facts LIMIT 0
                """)

                # Register the DataFrame and insert into temp table
                self.conn.register("temp_df", df)

                # Only insert columns that exist in the DataFrame
                df_columns = df.columns.tolist()
                common_columns = [col for col in column_names if col in df_columns]

                self.conn.execute(f"""
                    INSERT INTO temp_facts ({", ".join(common_columns)})
                    SELECT {", ".join(common_columns)} FROM temp_df
                """)

                # Insert or replace into the main table
                self.conn.execute(f"""
                    INSERT OR REPLACE INTO financial_facts ({", ".join(common_columns)})
                    SELECT {", ".join(common_columns)} FROM temp_facts
                """)

                # Drop the temporary table
                self.conn.execute("DROP TABLE IF EXISTS temp_facts")

                total_stored += len(batch)

            logger.info(f"Stored {total_stored} financial facts in batches")
            return total_stored
        except Exception as e:
            logger.error(f"Error storing financial facts batch: {e}")
            return 0

    def store_time_series_metric(self, metric_data: Dict[str, Any]) -> bool:
        """
        Store a single time series metric in the database.

        Args:
            metric_data: Dictionary with time series metric data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Store the metric using the batch method
            result = self.store_time_series_metrics_batch([metric_data])
            return result > 0
        except Exception as e:
            logger.error(f"Error storing time series metric: {e}")
            return False

    def store_time_series_metrics_batch(self, metrics_data: List[Dict[str, Any]]) -> int:
        """Store multiple time series metrics in a batch.

        Args:
            metrics_data: List of metric dictionaries

        Returns:
            Number of metrics stored
        """
        try:
            if not metrics_data:
                return 0

            # Prepare data for batch insert
            data = []
            for metric in metrics_data:
                ticker = metric.get("ticker")
                metric_name = metric.get("metric_name")
                fiscal_year = metric.get("fiscal_year")
                fiscal_quarter = metric.get("fiscal_quarter")
                value = metric.get("value")
                filing_id = metric.get("filing_id")

                if (
                    not ticker
                    or not metric_name
                    or not fiscal_year
                    or not fiscal_quarter
                    or value is None
                    or not filing_id
                ):
                    continue

                # Skip non-numeric values
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue

                row = {
                    "ticker": ticker,
                    "metric_name": metric_name,
                    "fiscal_year": fiscal_year,
                    "fiscal_quarter": fiscal_quarter,
                    "value": value,
                    "unit": metric.get("unit", "USD"),
                    "filing_id": filing_id,
                }
                data.append(row)

            if not data:
                return 0

            # Convert to DataFrame for batch insert
            df = pd.DataFrame(data)

            # Insert or replace metrics
            # Get the column names from the time_series_metrics table
            columns_result = self.conn.execute("PRAGMA table_info(time_series_metrics)").fetchdf()
            column_names = columns_result["name"].tolist()

            # Remove the created_at column as it has a default value
            if "created_at" in column_names:
                column_names.remove("created_at")

            # Create a temporary table with the same structure
            self.conn.execute(f"""
                CREATE TEMPORARY TABLE IF NOT EXISTS temp_metrics AS
                SELECT {", ".join(column_names)} FROM time_series_metrics LIMIT 0
            """)

            # Register the DataFrame and insert into temp table
            self.conn.register("temp_df", df)

            # Only insert columns that exist in the DataFrame
            df_columns = df.columns.tolist()
            common_columns = [col for col in column_names if col in df_columns]

            self.conn.execute(f"""
                INSERT INTO temp_metrics ({", ".join(common_columns)})
                SELECT {", ".join(common_columns)} FROM temp_df
            """)

            # Insert or replace into the main table
            self.conn.execute(f"""
                INSERT OR REPLACE INTO time_series_metrics ({", ".join(common_columns)})
                SELECT {", ".join(common_columns)} FROM temp_metrics
            """)

            # Drop the temporary table
            self.conn.execute("DROP TABLE IF EXISTS temp_metrics")

            logger.info(f"Stored {len(data)} time series metrics in batch")
            return len(data)
        except Exception as e:
            logger.error(f"Error storing time series metrics batch: {e}")
            return 0

    def store_financial_ratios_batch(self, ratios_data: List[Dict[str, Any]]) -> int:
        """Store multiple financial ratios in a batch.

        Args:
            ratios_data: List of ratio dictionaries

        Returns:
            Number of ratios stored
        """
        try:
            if not ratios_data:
                return 0

            # Prepare data for batch insert
            data = []
            for ratio in ratios_data:
                ticker = ratio.get("ticker")
                ratio_name = ratio.get("ratio_name")
                fiscal_year = ratio.get("fiscal_year")
                fiscal_quarter = ratio.get("fiscal_quarter")
                value = ratio.get("value")
                filing_id = ratio.get("filing_id")

                if (
                    not ticker
                    or not ratio_name
                    or not fiscal_year
                    or not fiscal_quarter
                    or value is None
                    or not filing_id
                ):
                    continue

                # Skip non-numeric values
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue

                row = {
                    "ticker": ticker,
                    "ratio_name": ratio_name,
                    "fiscal_year": fiscal_year,
                    "fiscal_quarter": fiscal_quarter,
                    "value": value,
                    "filing_id": filing_id,
                }
                data.append(row)

            if not data:
                return 0

            # Convert to DataFrame for batch insert
            df = pd.DataFrame(data)

            # Insert or replace ratios
            # Get the column names from the financial_ratios table
            columns_result = self.conn.execute("PRAGMA table_info(financial_ratios)").fetchdf()
            column_names = columns_result["name"].tolist()

            # Remove the created_at column as it has a default value
            if "created_at" in column_names:
                column_names.remove("created_at")

            # Create a temporary table with the same structure
            self.conn.execute(f"""
                CREATE TEMPORARY TABLE IF NOT EXISTS temp_ratios AS
                SELECT {", ".join(column_names)} FROM financial_ratios LIMIT 0
            """)

            # Register the DataFrame and insert into temp table
            self.conn.register("temp_df", df)

            # Only insert columns that exist in the DataFrame
            df_columns = df.columns.tolist()
            common_columns = [col for col in column_names if col in df_columns]

            self.conn.execute(f"""
                INSERT INTO temp_ratios ({", ".join(common_columns)})
                SELECT {", ".join(common_columns)} FROM temp_df
            """)

            # Insert or replace into the main table
            self.conn.execute(f"""
                INSERT OR REPLACE INTO financial_ratios ({", ".join(common_columns)})
                SELECT {", ".join(common_columns)} FROM temp_ratios
            """)

            # Drop the temporary table
            self.conn.execute("DROP TABLE IF EXISTS temp_ratios")

            logger.info(f"Stored {len(data)} financial ratios in batch")
            return len(data)
        except Exception as e:
            logger.error(f"Error storing financial ratios batch: {e}")
            return 0

    def store_financial_data_batch(self, financial_data: List[Dict[str, Any]]) -> int:
        """Store multiple financial data records in a batch.

        Args:
            financial_data: List of financial data dictionaries

        Returns:
            Number of records stored
        """
        try:
            if not financial_data:
                return 0

            # Filter out records with errors
            valid_data = [data for data in financial_data if "error" not in data]

            if not valid_data:
                return 0

            # Extract companies and filings
            companies = []
            filings = []
            all_facts = []
            all_metrics = []
            all_ratios = []

            for data in valid_data:
                # Extract company info
                ticker = data.get("ticker")
                if not ticker:
                    continue

                companies.append({"ticker": ticker})

                # Extract filing info
                filing_id = data.get("filing_id")
                accession_number = data.get("accession_number")
                fiscal_year = data.get("fiscal_year")
                fiscal_quarter = data.get("fiscal_quarter")

                if not filing_id or not accession_number or not fiscal_year or not fiscal_quarter:
                    continue

                filing = {
                    "id": filing_id,
                    "ticker": ticker,
                    "accession_number": accession_number,
                    "filing_type": data.get("filing_type"),
                    "filing_date": data.get("filing_date"),
                    "fiscal_year": fiscal_year,
                    "fiscal_quarter": fiscal_quarter,
                    "document_url": data.get("filing_url"),
                }
                filings.append(filing)

                # Extract facts
                facts = data.get("facts", [])
                for fact in facts:
                    fact["filing_id"] = filing_id
                    all_facts.append(fact)

                # Extract metrics
                metrics = data.get("metrics", {})
                for metric_name, value in metrics.items():
                    metric = {
                        "ticker": ticker,
                        "metric_name": metric_name,
                        "fiscal_year": fiscal_year,
                        "fiscal_quarter": fiscal_quarter,
                        "value": value,
                        "filing_id": filing_id,
                    }
                    all_metrics.append(metric)

                # Extract ratios
                ratios = data.get("ratios", {})
                for ratio_name, value in ratios.items():
                    ratio = {
                        "ticker": ticker,
                        "ratio_name": ratio_name,
                        "fiscal_year": fiscal_year,
                        "fiscal_quarter": fiscal_quarter,
                        "value": value,
                        "filing_id": filing_id,
                    }
                    all_ratios.append(ratio)

            # Store data in batches
            companies_stored = self.store_companies_batch(companies)
            filings_stored = self.store_filings_batch(filings)
            facts_stored = self.store_financial_facts_batch(all_facts)
            metrics_stored = self.store_time_series_metrics_batch(all_metrics)
            ratios_stored = self.store_financial_ratios_batch(all_ratios)

            logger.info(f"Stored {len(valid_data)} financial data records in batch")
            logger.info(f"  Companies: {companies_stored}")
            logger.info(f"  Filings: {filings_stored}")
            logger.info(f"  Facts: {facts_stored}")
            logger.info(f"  Metrics: {metrics_stored}")
            logger.info(f"  Ratios: {ratios_stored}")

            return len(valid_data)
        except Exception as e:
            logger.error(f"Error storing financial data batch: {e}")
            return 0

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
