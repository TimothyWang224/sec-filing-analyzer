"""
DuckDB UI Launcher

Utility functions for launching the DuckDB UI.
"""

import logging
import os
import webbrowser
from pathlib import Path

import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)


def launch_duckdb_ui(db_path: str = None):
    """
    Launch the DuckDB UI in a new browser tab.

    Args:
        db_path: Path to the DuckDB database file. If None, uses the default path.
    """
    try:
        # Default database path if not provided
        if db_path is None:
            db_path = "data/db_backup/improved_financial_data.duckdb"

        # Make sure the path is absolute
        db_path = os.path.abspath(db_path)

        # Check if the database file exists
        if not os.path.exists(db_path):
            st.error(f"Database file not found: {db_path}")
            logger.error(f"Database file not found: {db_path}")
            return

        # Construct the URL for DuckDB UI
        # This assumes you're using the DuckDB Web UI or a similar tool
        # You may need to adjust this URL based on your specific setup
        url = f"file://{db_path}"

        # For now, just show a message since we don't have a direct web UI for DuckDB
        st.info(f"Opening DuckDB database: {db_path}")
        st.info(
            "Note: DuckDB Web UI is not directly available. You can use DBeaver or another SQL client to connect to this database."
        )

        # Log the action
        logger.info(f"Attempted to launch DuckDB UI for database: {db_path}")

        # Return the path for potential use elsewhere
        return db_path

    except Exception as e:
        st.error(f"Error launching DuckDB UI: {e}")
        logger.error(f"Error launching DuckDB UI: {e}")
        return None
