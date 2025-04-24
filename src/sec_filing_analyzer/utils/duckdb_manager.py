"""
DuckDB Connection Manager

This module provides a centralized way to manage DuckDB connections,
with support for read-only and read-write modes.
"""

import logging
import os
from typing import Optional
import duckdb
from pathlib import Path

logger = logging.getLogger(__name__)

class DuckDBManager:
    """
    A manager for DuckDB connections that supports read-only and read-write modes.

    This class helps prevent database locking issues by defaulting to read-only
    connections for most operations, with write access only when explicitly needed.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(DuckDBManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, default_db_path: Optional[str] = None):
        """Initialize the DuckDB manager.

        Args:
            default_db_path: Default path to the DuckDB database file
        """
        if self._initialized:
            return

        self.default_db_path = default_db_path or "data/db_backup/financial_data.duckdb"
        self._active_connections = {}
        self._initialized = True
        logger.info(f"Initialized DuckDB manager with default path: {self.default_db_path}")

    def get_connection(self, db_path: Optional[str] = None, read_only: bool = True) -> duckdb.DuckDBPyConnection:
        """Get a DuckDB connection.

        Args:
            db_path: Path to the DuckDB database file (uses default if None)
            read_only: Whether to open the connection in read-only mode

        Returns:
            A DuckDB connection
        """
        db_path = db_path or self.default_db_path

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Check if we already have any connection to this database
        read_only_key = f"{db_path}:True"
        read_write_key = f"{db_path}:False"

        # If we have a read-write connection and are requesting read-only, use the read-write connection
        if read_only and read_write_key in self._active_connections:
            logger.info(f"Using existing read-write connection for {db_path} instead of creating a read-only connection")
            return self._active_connections[read_write_key]

        # If we have a read-only connection and are requesting read-write, close it first
        if not read_only and read_only_key in self._active_connections:
            logger.info(f"Closing existing read-only connection to {db_path} before creating read-write connection")
            self._active_connections[read_only_key].close()
            del self._active_connections[read_only_key]

        # Generate a connection key based on path and mode
        conn_key = f"{db_path}:{read_only}"

        # Return existing connection if available
        if conn_key in self._active_connections:
            return self._active_connections[conn_key]

        # Create new connection
        try:
            conn = duckdb.connect(db_path, read_only=read_only)
            self._active_connections[conn_key] = conn
            mode_str = "read-only" if read_only else "read-write"
            logger.info(f"Connected to DuckDB at {db_path} in {mode_str} mode")
            return conn
        except Exception as e:
            logger.error(f"Error connecting to DuckDB at {db_path}: {e}")
            # If we failed to create a read-only connection, try with read-write as a fallback
            if read_only:
                try:
                    logger.info(f"Attempting to connect to {db_path} in read-write mode as fallback")
                    conn = duckdb.connect(db_path, read_only=False)
                    self._active_connections[f"{db_path}:False"] = conn
                    logger.info(f"Connected to DuckDB at {db_path} in read-write mode (fallback)")
                    return conn
                except Exception as inner_e:
                    logger.error(f"Fallback connection also failed: {inner_e}")
            raise

    def get_read_only_connection(self, db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
        """Get a read-only DuckDB connection.

        Args:
            db_path: Path to the DuckDB database file (uses default if None)

        Returns:
            A read-only DuckDB connection
        """
        return self.get_connection(db_path, read_only=True)

    def get_read_write_connection(self, db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
        """Get a read-write DuckDB connection.

        Args:
            db_path: Path to the DuckDB database file (uses default if None)

        Returns:
            A read-write DuckDB connection
        """
        return self.get_connection(db_path, read_only=False)

    def close_connection(self, db_path: Optional[str] = None, read_only: Optional[bool] = None):
        """Close a specific DuckDB connection.

        Args:
            db_path: Path to the DuckDB database file (uses default if None)
            read_only: Whether the connection is read-only (closes both if None)
        """
        db_path = db_path or self.default_db_path

        if read_only is None:
            # Close both read-only and read-write connections
            ro_key = f"{db_path}:True"
            rw_key = f"{db_path}:False"

            if ro_key in self._active_connections:
                self._active_connections[ro_key].close()
                del self._active_connections[ro_key]
                logger.info(f"Closed read-only connection to {db_path}")

            if rw_key in self._active_connections:
                self._active_connections[rw_key].close()
                del self._active_connections[rw_key]
                logger.info(f"Closed read-write connection to {db_path}")
        else:
            # Close specific connection
            conn_key = f"{db_path}:{read_only}"
            if conn_key in self._active_connections:
                self._active_connections[conn_key].close()
                del self._active_connections[conn_key]
                mode_str = "read-only" if read_only else "read-write"
                logger.info(f"Closed {mode_str} connection to {db_path}")

    def close_all_connections(self):
        """Close all active DuckDB connections."""
        for conn_key, conn in list(self._active_connections.items()):
            try:
                conn.close()
                logger.info(f"Closed connection: {conn_key}")
            except Exception as e:
                logger.warning(f"Error closing connection {conn_key}: {e}")
        self._active_connections.clear()
        logger.info("Closed all DuckDB connections")

    def force_close_all_connections(self, db_path: Optional[str] = None):
        """Force close all connections to a database by creating a new connection and closing it.

        This is a more aggressive approach that can help when normal connection closing fails.

        Args:
            db_path: Path to the DuckDB database file (uses default if None)
        """
        db_path = db_path or self.default_db_path

        # First try normal closing
        self.close_connection(db_path)

        # Clear our connection tracking for this database
        ro_key = f"{db_path}:True"
        rw_key = f"{db_path}:False"
        if ro_key in self._active_connections:
            try:
                self._active_connections[ro_key].close()
            except Exception as e:
                logger.warning(f"Error closing read-only connection: {e}")
            del self._active_connections[ro_key]
            logger.info(f"Removed read-only connection from tracking: {db_path}")

        if rw_key in self._active_connections:
            try:
                self._active_connections[rw_key].close()
            except Exception as e:
                logger.warning(f"Error closing read-write connection: {e}")
            del self._active_connections[rw_key]
            logger.info(f"Removed read-write connection from tracking: {db_path}")

        # Then try to force close by creating a new exclusive connection
        try:
            # Create a direct connection outside our connection pool
            logger.info(f"Attempting to force close connections to {db_path}")

            # Make sure the directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            # Try to connect with exclusive access
            temp_conn = duckdb.connect(db_path, read_only=False)
            temp_conn.close()
            logger.info(f"Successfully forced close of connections to {db_path}")

        except Exception as e:
            logger.warning(f"Could not force close connections to {db_path}: {e}")

    def get_database_info(self, db_path: Optional[str] = None) -> dict:
        """Get information about the database.

        Args:
            db_path: Path to the DuckDB database file (uses default if None)

        Returns:
            A dictionary with database information
        """
        db_path = db_path or self.default_db_path

        # Force close existing connections to ensure we can access the database
        self.force_close_all_connections(db_path)

        result = {
            "path": db_path,
            "exists": os.path.exists(db_path),
            "size_mb": round(os.path.getsize(db_path) / (1024 * 1024), 2) if os.path.exists(db_path) else 0,
            "tables": [],
            "row_counts": {},
            "version": None,
            "error": None
        }

        # If the database doesn't exist, return early
        if not result["exists"]:
            result["error"] = f"Database file not found at {db_path}"
            return result

        # Try to get database information using direct SQL queries
        try:
            # Create a new connection with a timeout
            import time
            max_attempts = 3
            attempt = 0
            conn = None

            while attempt < max_attempts and conn is None:
                try:
                    # Try to connect with read-only access first
                    conn = duckdb.connect(db_path, read_only=True)
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}/{max_attempts} to connect to {db_path} failed: {e}")
                    attempt += 1
                    if attempt < max_attempts:
                        # Wait before retrying
                        time.sleep(1)
                        # Try to force close connections again
                        self.force_close_all_connections(db_path)

            if conn is None:
                # If we still can't connect, try with read-write as a last resort
                try:
                    conn = duckdb.connect(db_path, read_only=False)
                    logger.info(f"Connected to {db_path} in read-write mode as fallback")
                except Exception as e:
                    result["error"] = f"Failed to connect to database after {max_attempts} attempts: {e}"
                    return result

            # Get DuckDB version
            result["version"] = conn.execute("SELECT version()").fetchone()[0]

            # Get table list
            tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name").fetchall()
            result["tables"] = [t[0] for t in tables]

            # Get row counts for each table
            for table in result["tables"]:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    result["row_counts"][table] = count
                except Exception as e:
                    result["row_counts"][table] = f"Error: {e}"

            # Close the connection
            conn.close()

        except Exception as e:
            result["error"] = str(e)

        return result

# Create a global instance
duckdb_manager = DuckDBManager()
