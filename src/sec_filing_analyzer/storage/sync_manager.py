"""
Synchronization Manager for SEC Filing Analyzer

This module provides functionality to synchronize data between different storage systems:
- DuckDB (relational database)
- Vector Store (embeddings)
- File System (raw filings, processed filings, etc.)
- Neo4j (graph database)
"""

import os
import logging
import duckdb
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set

import numpy as np

# Import the DuckDB manager
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.sec_filing_analyzer.utils.duckdb_manager import duckdb_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageSyncManager:
    """
    Manager for synchronizing data between different storage systems.
    """

    def __init__(
        self,
        db_path: str = "data/financial_data.duckdb",
        vector_store_path: str = "data/vector_store",
        filings_dir: str = "data/filings",
        graph_store_dir: str = "data/graph_store",
        read_only: bool = True
    ):
        """
        Initialize the storage synchronization manager.

        Args:
            db_path: Path to the DuckDB database
            vector_store_path: Path to the vector store
            filings_dir: Path to the filings directory
            graph_store_dir: Path to the graph store directory
            read_only: Whether to open the database in read-only mode
        """
        self.db_path = db_path
        self.vector_store_path = Path(vector_store_path)
        self.filings_dir = Path(filings_dir)
        self.graph_store_dir = Path(graph_store_dir)
        self.read_only = read_only

        # Initialize DuckDB connection
        self._init_db_connection()

    def _init_db_connection(self):
        """Initialize the DuckDB connection."""
        try:
            # Use the DuckDB manager to get a connection with the appropriate mode
            if self.read_only:
                self.conn = duckdb_manager.get_read_only_connection(self.db_path)
                logger.info(f"Connected to DuckDB at {self.db_path} in read-only mode")
            else:
                self.conn = duckdb_manager.get_read_write_connection(self.db_path)
                logger.info(f"Connected to DuckDB at {self.db_path} in read-write mode")
        except Exception as e:
            logger.error(f"Error connecting to DuckDB: {e}")
            raise

    def close(self):
        """Close the DuckDB connection."""
        # We don't actually close the connection since it's managed by the DuckDB manager
        # The DuckDB manager will handle connection pooling and closing when appropriate
        logger.info("Connection managed by DuckDB manager - not explicitly closing")

    def sync_all(self) -> Dict[str, Any]:
        """
        Synchronize all storage systems.

        Returns:
            Dictionary with synchronization results
        """
        results = {
            "vector_store": {"found": 0, "added": 0, "updated": 0, "errors": 0},
            "file_system": {"found": 0, "added": 0, "updated": 0, "errors": 0},
            "total_filings": 0
        }

        # Sync vector store
        vs_results = self.sync_vector_store()
        results["vector_store"] = vs_results

        # Sync file system
        fs_results = self.sync_file_system()
        results["file_system"] = fs_results

        # Get total filings count
        total = self.conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
        results["total_filings"] = total

        return results

    def sync_vector_store(self) -> Dict[str, int]:
        """
        Synchronize vector store with DuckDB.

        Returns:
            Dictionary with synchronization results
        """
        results = {"found": 0, "added": 0, "updated": 0, "errors": 0}

        try:
            # Check if vector store exists
            if not self.vector_store_path.exists():
                logger.warning(f"Vector store path {self.vector_store_path} does not exist")
                return results

            # Check if company directory exists
            company_dir = self.vector_store_path / "by_company"
            if not company_dir.exists():
                logger.warning(f"Company directory {company_dir} does not exist")
                return results

            # Get all companies in the vector store
            companies = [d.name for d in company_dir.iterdir() if d.is_dir()]
            logger.info(f"Found {len(companies)} companies in vector store")

            # Process each company
            for ticker in companies:
                # Skip test companies
                if ticker.startswith("TEST"):
                    continue

                company_path = company_dir / ticker

                # Get all embedding files for this company
                embedding_files = list(company_path.glob("*.npy"))
                logger.info(f"Found {len(embedding_files)} embedding files for {ticker}")

                # Track accession numbers found
                accession_numbers = set()

                # Process each embedding file
                for emb_file in embedding_files:
                    try:
                        # Extract accession number from filename
                        filename = emb_file.stem

                        # Skip chunk embeddings for now, focus on document embeddings
                        if "_chunk_" in filename:
                            continue

                        # Extract accession number
                        accession_number = filename
                        accession_numbers.add(accession_number)

                        results["found"] += 1

                        # Check if this filing is already in DuckDB
                        existing = self.conn.execute(
                            "SELECT filing_id, document_url, fiscal_period FROM filings WHERE accession_number = ?",
                            [accession_number]
                        ).fetchone()

                        if existing:
                            # Update fiscal period if needed
                            if existing[2] != "Q3":  # Q3 represents embedded status
                                self.conn.execute(
                                    """
                                    UPDATE filings
                                    SET fiscal_period = 'Q3', updated_at = CURRENT_TIMESTAMP
                                    WHERE accession_number = ?
                                    """,
                                    [accession_number]
                                )
                                results["updated"] += 1
                        else:
                            # Try to get metadata from vector store
                            metadata_path = self.vector_store_path / "metadata" / f"{filename}.json"
                            if metadata_path.exists():
                                with open(metadata_path, "r") as f:
                                    metadata = json.load(f)

                                # Extract filing information
                                filing_type = metadata.get("filing_type", "unknown")
                                filing_date = metadata.get("filing_date", "2000-01-01")

                                # Generate a filing ID
                                filing_id = f"{ticker}_{accession_number}"

                                # Get company_id
                                company_id = self._get_company_id(ticker)

                                # Add to DuckDB
                                self.conn.execute(
                                    """
                                    INSERT INTO filings (
                                        filing_id, company_id, accession_number, filing_type, filing_date,
                                        fiscal_period, created_at, updated_at
                                    ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                                    """,
                                    [len(str(filing_id)), company_id, accession_number, filing_type, filing_date, "Q3"]
                                )

                                # Make sure company exists
                                self._ensure_company_exists(ticker)

                                results["added"] += 1
                            else:
                                # No metadata, create minimal record
                                filing_id = f"{ticker}_{accession_number}"

                                # Get company_id
                                company_id = self._get_company_id(ticker)

                                # Add to DuckDB with minimal information
                                self.conn.execute(
                                    """
                                    INSERT INTO filings (
                                        filing_id, company_id, accession_number, fiscal_period, created_at, updated_at
                                    ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                                    """,
                                    [len(str(filing_id)), company_id, accession_number, "Q3"]
                                )

                                # Make sure company exists
                                self._ensure_company_exists(ticker)

                                results["added"] += 1

                    except Exception as e:
                        logger.error(f"Error processing embedding file {emb_file}: {e}")
                        results["errors"] += 1

            return results

        except Exception as e:
            logger.error(f"Error synchronizing vector store: {e}")
            results["errors"] += 1
            return results

    def sync_file_system(self) -> Dict[str, int]:
        """
        Synchronize file system with DuckDB.

        Returns:
            Dictionary with synchronization results
        """
        results = {"found": 0, "added": 0, "updated": 0, "errors": 0}

        try:
            # Check if filings directory exists
            if not self.filings_dir.exists():
                logger.warning(f"Filings directory {self.filings_dir} does not exist")
                return results

            # Check different filing directories
            for subdir in ["raw", "html", "processed", "xbrl"]:
                subdir_path = self.filings_dir / subdir
                if not subdir_path.exists():
                    continue

                # Get all companies in this directory
                companies = [d.name for d in subdir_path.iterdir() if d.is_dir()]
                logger.info(f"Found {len(companies)} companies in {subdir}")

                # Process each company
                for ticker in companies:
                    company_path = subdir_path / ticker

                    # Get all years (if organized by year)
                    years = [d.name for d in company_path.iterdir() if d.is_dir()]

                    if years:
                        # Process each year
                        for year in years:
                            year_path = company_path / year
                            self._process_filing_directory(ticker, year_path, results, subdir)
                    else:
                        # Process company directory directly
                        self._process_filing_directory(ticker, company_path, results, subdir)

            return results

        except Exception as e:
            logger.error(f"Error synchronizing file system: {e}")
            results["errors"] += 1
            return results

    def _process_filing_directory(self, ticker: str, directory: Path, results: Dict[str, int], file_type: str):
        """
        Process a directory containing filings.

        Args:
            ticker: Company ticker
            directory: Directory to process
            results: Results dictionary to update
            file_type: Type of files (raw, html, processed, xbrl)
        """
        try:
            # Get all files in this directory
            files = list(directory.glob("*.*"))

            # Skip metadata files for now
            files = [f for f in files if not f.name.endswith("_metadata.json")]

            logger.info(f"Found {len(files)} files for {ticker} in {directory}")

            # Process each file
            for file_path in files:
                try:
                    # Extract accession number from filename
                    filename = file_path.stem

                    # Try to extract accession number
                    accession_number = self._extract_accession_number(filename)

                    if accession_number:
                        results["found"] += 1

                        # Check if this filing is already in DuckDB
                        existing = self.conn.execute(
                            "SELECT filing_id, document_url, fiscal_period FROM filings WHERE accession_number = ?",
                            [accession_number]
                        ).fetchone()

                        if existing:
                            # Update document URL if needed
                            if not existing[1]:
                                self.conn.execute(
                                    """
                                    UPDATE filings
                                    SET document_url = ?, updated_at = CURRENT_TIMESTAMP
                                    WHERE accession_number = ?
                                    """,
                                    [str(file_path), accession_number]
                                )
                                results["updated"] += 1
                        else:
                            # Try to extract filing type and date from filename or path
                            filing_type, filing_date = self._extract_filing_info(file_path, ticker)

                            # Get company_id
                            company_id = self._get_company_id(ticker)

                            # Generate a filing ID
                            max_id = self.conn.execute("SELECT MAX(filing_id) FROM filings").fetchone()[0]
                            filing_id = 1 if max_id is None else max_id + 1

                            # Add to DuckDB
                            self.conn.execute(
                                """
                                INSERT INTO filings (
                                    filing_id, company_id, accession_number, filing_type, filing_date,
                                    document_url, fiscal_period, created_at, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                                """,
                                [filing_id, company_id, accession_number, filing_type, filing_date,
                                 str(file_path), "Q1"]
                            )

                            # Make sure company exists
                            self._ensure_company_exists(ticker)

                            results["added"] += 1

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    results["errors"] += 1

        except Exception as e:
            logger.error(f"Error processing directory {directory}: {e}")
            results["errors"] += 1

    def _extract_accession_number(self, filename: str) -> Optional[str]:
        """
        Extract accession number from filename.

        Args:
            filename: Filename to extract from

        Returns:
            Accession number or None if not found
        """
        # Check if filename is already an accession number format (0000XXXXXX-YY-NNNNNN)
        if "-" in filename and len(filename) >= 18:
            parts = filename.split("-")
            if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                return filename

        # Try to extract from common formats
        if "_" in filename:
            parts = filename.split("_")
            for part in parts:
                if "-" in part and len(part) >= 18:
                    subparts = part.split("-")
                    if len(subparts) == 3 and subparts[0].isdigit() and subparts[1].isdigit() and subparts[2].isdigit():
                        return part

        # No accession number found
        return None

    def _extract_filing_info(self, file_path: Path, ticker: str) -> Tuple[str, str]:
        """
        Extract filing type and date from file path.

        Args:
            file_path: Path to the filing file
            ticker: Company ticker

        Returns:
            Tuple of (filing_type, filing_date)
        """
        filing_type = "unknown"
        filing_date = "2000-01-01"

        # Try to extract from path components
        path_parts = file_path.parts

        # Check if filing type is in the path
        for filing_type_candidate in ["10-K", "10-Q", "8-K", "20-F", "40-F", "6-K", "DEF 14A"]:
            if filing_type_candidate in path_parts:
                filing_type = filing_type_candidate
                break

        # Try to extract date from filename
        filename = file_path.stem

        # Check if filename is an accession number
        if "-" in filename and len(filename) >= 18:
            parts = filename.split("-")
            if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                # This is an accession number, try to extract year from it
                # Format: 0000XXXXXX-YY-NNNNNN
                year = "20" + parts[1]  # Assuming all filings are from 2000 onwards
                # Use a default month and day
                filing_date = f"{year}-01-01"
                return filing_type, filing_date

        # Look for date patterns in the filename
        date_patterns = [
            # YYYY-MM-DD
            r"(\d{4}-\d{2}-\d{2})",
            # YYYYMMDD
            r"(\d{8})",
            # YYYY_MM_DD
            r"(\d{4}_\d{2}_\d{2})"
        ]

        import re
        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1)
                # Normalize date format
                if len(date_str) == 8 and date_str.isdigit():
                    # YYYYMMDD
                    filing_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                elif "_" in date_str:
                    # YYYY_MM_DD
                    filing_date = date_str.replace("_", "-")
                else:
                    filing_date = date_str
                break

        # Check if we have a year directory in the path
        for part in path_parts:
            if part.isdigit() and len(part) == 4 and 1990 <= int(part) <= 2100:
                # This is a year directory, use it if we don't have a better date
                if filing_date == "2000-01-01":
                    filing_date = f"{part}-01-01"
                break

        return filing_type, filing_date

    def _get_company_id(self, ticker: str) -> int:
        """
        Get the company ID for a ticker, creating the company if it doesn't exist.

        Args:
            ticker: Company ticker

        Returns:
            Company ID
        """
        # Check if company exists
        existing = self.conn.execute(
            "SELECT company_id FROM companies WHERE ticker = ?",
            [ticker]
        ).fetchone()

        if existing:
            return existing[0]
        else:
            # Add company
            try:
                # Get the next available company_id
                max_id = self.conn.execute("SELECT MAX(company_id) FROM companies").fetchone()[0]
                company_id = 1 if max_id is None else max_id + 1

                self.conn.execute(
                    """INSERT INTO companies (
                        company_id, ticker, name, created_at, updated_at
                    ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                    [company_id, ticker, ticker]
                )
                logger.info(f"Added company {ticker} (ID: {company_id}) to database")
                return company_id
            except Exception as e:
                logger.error(f"Error adding company {ticker} to database: {e}")
                # Try a different approach - disable foreign key constraints temporarily
                try:
                    self.conn.execute("PRAGMA foreign_keys = OFF")
                    # Get the next available company_id
                    max_id = self.conn.execute("SELECT MAX(company_id) FROM companies").fetchone()[0]
                    company_id = 1 if max_id is None else max_id + 1

                    self.conn.execute(
                        """INSERT INTO companies (
                            company_id, ticker, name, created_at, updated_at
                        ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                        [company_id, ticker, ticker]
                    )
                    self.conn.execute("PRAGMA foreign_keys = ON")
                    logger.info(f"Added company {ticker} (ID: {company_id}) to database with foreign keys disabled")
                    return company_id
                except Exception as e2:
                    logger.error(f"Error adding company {ticker} to database with foreign keys disabled: {e2}")
                    # Return a default ID as a last resort
                    return 999999

    def _ensure_company_exists(self, ticker: str):
        """
        Ensure a company exists in the database.

        Args:
            ticker: Company ticker
        """
        self._get_company_id(ticker)

    def update_filing_paths(self) -> Dict[str, int]:
        """
        Update file paths for filings in DuckDB.

        Returns:
            Dictionary with update results
        """
        results = {"updated": 0, "errors": 0, "not_found": 0}

        try:
            # Get all filings without a local file path
            filings = self.conn.execute(
                """
                SELECT
                    CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'id')
                        THEN id
                        ELSE filing_id
                    END as id,
                    CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'ticker')
                        THEN ticker
                        ELSE (SELECT c.ticker FROM companies c WHERE c.company_id = filings.company_id)
                    END as ticker,
                    accession_number
                FROM filings
                WHERE (
                    (EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'local_file_path') AND local_file_path IS NULL) OR
                    (NOT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'local_file_path'))
                )
                """
            ).fetchdf()

            if filings.empty:
                logger.info("No filings without local file path")
                return results

            logger.info(f"Found {len(filings)} filings without local file path")

            # Process each filing
            for _, filing in filings.iterrows():
                ticker = filing['ticker']
                accession_number = filing['accession_number']

                # Try to find the file
                file_path = self._find_filing_file(ticker, accession_number)

                if file_path:
                    # Update the filing
                    self.conn.execute(
                        """
                        UPDATE filings
                        SET local_file_path = ?, last_updated = CURRENT_TIMESTAMP
                        WHERE accession_number = ?
                        """,
                        [str(file_path), accession_number]
                    )
                    results["updated"] += 1
                else:
                    results["not_found"] += 1

            return results

        except Exception as e:
            logger.error(f"Error updating filing paths: {e}")
            results["errors"] += 1
            return results

    def _find_filing_file(self, ticker: str, accession_number: str) -> Optional[Path]:
        """
        Find a filing file in the file system.

        Args:
            ticker: Company ticker
            accession_number: SEC accession number

        Returns:
            Path to the filing file or None if not found
        """
        # Check different filing directories
        for subdir in ["raw", "html", "processed", "xbrl"]:
            subdir_path = self.filings_dir / subdir
            if not subdir_path.exists():
                continue

            # Check company directory
            company_path = subdir_path / ticker
            if not company_path.exists():
                continue

            # Search for the file
            for ext in ["html", "txt", "xml", "json"]:
                # Try direct match
                file_path = company_path / f"{accession_number}.{ext}"
                if file_path.exists():
                    return file_path

                # Try in year subdirectories
                for year_dir in company_path.iterdir():
                    if year_dir.is_dir():
                        file_path = year_dir / f"{accession_number}.{ext}"
                        if file_path.exists():
                            return file_path

                # Try with ticker prefix
                file_path = company_path / f"{ticker}_{accession_number}.{ext}"
                if file_path.exists():
                    return file_path

                # Try in year subdirectories with ticker prefix
                for year_dir in company_path.iterdir():
                    if year_dir.is_dir():
                        file_path = year_dir / f"{ticker}_{accession_number}.{ext}"
                        if file_path.exists():
                            return file_path

        # File not found
        return None

    def update_processing_status(self) -> Dict[str, int]:
        """
        Update fiscal period for filings in DuckDB (equivalent to processing status in old schema).

        Returns:
            Dictionary with update results
        """
        results = {"updated": 0, "errors": 0}

        try:
            # Get all filings with null fiscal period
            filings = self.conn.execute(
                """
                SELECT
                    CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'filing_id')
                        THEN filing_id
                        ELSE id
                    END as filing_id,
                    accession_number
                FROM filings
                WHERE (
                    (EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'fiscal_period') AND fiscal_period IS NULL) OR
                    (EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'fiscal_period') AND fiscal_period IS NULL AND fiscal_quarter IS NOT NULL) OR
                    (NOT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'fiscal_period'))
                )
                """
            ).fetchdf()

            if filings.empty:
                logger.info("No filings with unknown fiscal period")
                return results

            logger.info(f"Found {len(filings)} filings with unknown fiscal period")

            # Process each filing
            for _, filing in filings.iterrows():
                # Get company ticker from filing
                company_info = self.conn.execute(
                    """
                    SELECT c.ticker
                    FROM filings f
                    JOIN companies c ON
                        CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'company_id')
                            THEN f.company_id = c.company_id
                            ELSE f.ticker = c.ticker
                        END
                    WHERE
                        CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'filing_id')
                            THEN f.filing_id = ?
                            ELSE f.id = ?
                        END
                    """,
                    [filing['filing_id'], filing['filing_id']]
                ).fetchone()

                if not company_info:
                    logger.warning(f"Could not find company for filing {filing['filing_id']}")
                    continue

                ticker = company_info[0]
                accession_number = filing['accession_number']

                # Determine processing status
                status = self._determine_processing_status(ticker, accession_number)

                # Map status to fiscal period
                status_map = {
                    "downloaded": "Q1",
                    "processed": "Q2",
                    "embedded": "Q3",
                    "xbrl_processed": "Q4",
                    "error": "FY",
                    "unknown": None
                }
                fiscal_period = status_map.get(status, None)

                # Update the filing
                self.conn.execute(
                    """
                    UPDATE filings
                    SET fiscal_period = ?,
                        updated_at = CASE
                            WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'updated_at')
                                THEN CASE WHEN updated_at IS NOT NULL THEN CURRENT_TIMESTAMP ELSE created_at END
                            ELSE NULL
                        END
                    WHERE
                        CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'filing_id')
                            THEN filing_id = ?
                            ELSE id = ?
                        END
                    """,
                    [fiscal_period, filing['filing_id'], filing['filing_id']]
                )

                results["updated"] += 1

            return results

        except Exception as e:
            logger.error(f"Error updating fiscal period: {e}")
            results["errors"] += 1
            return results

    def _determine_processing_status(self, ticker: str, accession_number: str) -> str:
        """
        Determine the processing status of a filing.

        Args:
            ticker: Company ticker
            accession_number: SEC accession number

        Returns:
            Processing status
        """
        # Check if file exists
        file_path = self._find_filing_file(ticker, accession_number)
        if not file_path:
            return "missing"

        # Check if embedding exists
        embedding_path = self.vector_store_path / "by_company" / ticker / f"{accession_number}.npy"
        if embedding_path.exists():
            return "embedded"

        # Check if XBRL data exists
        xbrl_path = self.filings_dir / "xbrl" / ticker / f"{accession_number}.json"
        if xbrl_path.exists():
            return "xbrl_processed"

        # Check if processed data exists
        processed_path = self.filings_dir / "processed" / ticker / f"{accession_number}.json"
        if processed_path.exists():
            return "processed"

        # Default to downloaded
        return "downloaded"

    def get_inventory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the filing inventory.

        Returns:
            Dictionary with inventory summary
        """
        try:
            # Get total filings count
            total = self.conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]

            # Get counts by fiscal period (mapped to processing status for compatibility)
            status_counts = self.conn.execute("""
                SELECT fiscal_period, COUNT(*) as count
                FROM filings
                GROUP BY fiscal_period
            """).fetchdf()

            # Map fiscal periods to status names for compatibility
            status_map = {
                "Q1": "downloaded",
                "Q2": "processed",
                "Q3": "embedded",
                "Q4": "xbrl_processed",
                "FY": "completed",
                None: "unknown"
            }

            # Transform the status counts
            status_counts_records = []
            for _, row in status_counts.iterrows():
                fiscal_period = row['fiscal_period']
                status_name = status_map.get(fiscal_period, "unknown")
                status_counts_records.append({
                    "processing_status": status_name,
                    "count": row['count']
                })

            # Get counts by company
            company_counts = self.conn.execute("""
                SELECT c.ticker, COUNT(*) as count
                FROM filings f
                JOIN companies c ON f.company_id = c.company_id
                GROUP BY c.ticker
                ORDER BY count DESC
            """).fetchdf()

            # Get counts by filing type
            type_counts = self.conn.execute("""
                SELECT filing_type, COUNT(*) as count
                FROM filings
                GROUP BY filing_type
                ORDER BY count DESC
            """).fetchdf()

            # Get counts by year
            year_counts = self.conn.execute("""
                SELECT EXTRACT(YEAR FROM filing_date) as year, COUNT(*) as count
                FROM filings
                GROUP BY year
                ORDER BY year DESC
            """).fetchdf()

            return {
                "total_filings": total,
                "status_counts": status_counts_records,
                "company_counts": company_counts.to_dict(orient="records"),
                "type_counts": type_counts.to_dict(orient="records"),
                "year_counts": year_counts.to_dict(orient="records")
            }

        except Exception as e:
            logger.error(f"Error getting inventory summary: {e}")
            return {"error": str(e)}


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Storage Synchronization Manager")
    parser.add_argument("--db-path", default="data/financial_data.duckdb", help="Path to DuckDB database")
    parser.add_argument("--vector-store-path", default="data/vector_store", help="Path to vector store")
    parser.add_argument("--filings-dir", default="data/filings", help="Path to filings directory")
    parser.add_argument("--graph-store-dir", default="data/graph_store", help="Path to graph store directory")
    parser.add_argument("--action", choices=["sync-all", "sync-vector-store", "sync-file-system",
                                            "update-paths", "update-status", "summary"],
                        default="sync-all", help="Action to perform")

    args = parser.parse_args()

    # Create sync manager
    sync_manager = StorageSyncManager(
        db_path=args.db_path,
        vector_store_path=args.vector_store_path,
        filings_dir=args.filings_dir,
        graph_store_dir=args.graph_store_dir
    )

    # Perform action
    if args.action == "sync-all":
        results = sync_manager.sync_all()
        print(json.dumps(results, indent=2))
    elif args.action == "sync-vector-store":
        results = sync_manager.sync_vector_store()
        print(json.dumps(results, indent=2))
    elif args.action == "sync-file-system":
        results = sync_manager.sync_file_system()
        print(json.dumps(results, indent=2))
    elif args.action == "update-paths":
        results = sync_manager.update_filing_paths()
        print(json.dumps(results, indent=2))
    elif args.action == "update-status":
        results = sync_manager.update_processing_status()
        print(json.dumps(results, indent=2))
    elif args.action == "summary":
        results = sync_manager.get_inventory_summary()
        print(json.dumps(results, indent=2))

    # Close connection
    sync_manager.close()
