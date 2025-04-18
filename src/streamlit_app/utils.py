"""
Utility functions for the Streamlit application.
"""

import streamlit as st
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the DuckDB manager
from src.sec_filing_analyzer.utils.duckdb_manager import duckdb_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_utils")

def launch_duckdb_ui(db_path="data/financial_data.duckdb"):
    """
    Launch the DuckDB UI in a new browser tab.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Note: We need a read-write connection to launch the UI
        logger.info(f"Connecting to DuckDB database at {db_path}")
        conn = duckdb_manager.get_read_write_connection(db_path)
        logger.info("Launching DuckDB UI")
        conn.execute("CALL start_ui()")
        st.success("DuckDB UI launched! Check your browser for a new tab.")
        return True
    except Exception as e:
        logger.error(f"Error launching DuckDB UI: {e}")
        st.error(f"Error launching DuckDB UI: {e}")
        st.info("Make sure you have DuckDB v1.2.1 or newer installed.")
        return False
