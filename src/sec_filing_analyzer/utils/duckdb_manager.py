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

        self.default_db_path = default_db_path or "data/db_backup/improved_financial_data.duckdb"
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
            conn.close()
            logger.info(f"Closed connection: {conn_key}")
        self._active_connections.clear()
        logger.info("Closed all DuckDB connections")

# Create a global instance
duckdb_manager = DuckDBManager()
