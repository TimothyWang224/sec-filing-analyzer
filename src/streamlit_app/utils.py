"""
Utility functions for the Streamlit application.
"""

import logging
import os
import sys
from pathlib import Path

import streamlit as st

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the DuckDB manager
from src.sec_filing_analyzer.utils.duckdb_manager import duckdb_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_utils")


def launch_duckdb_ui(db_path="data/db_backup/improved_financial_data.duckdb"):
    """
    Launch the native DuckDB UI in a new browser tab using DuckDB's built-in GUI functionality.
    For DuckDB versions that don't support the native GUI (pre-0.9.0), provide instructions for alternatives.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        True if successful, False otherwise
    """
    # Create a dedicated logger for DuckDB UI operations
    duckdb_ui_logger = logging.getLogger("src.streamlit_app.utils.duckdb_ui")

    # Get absolute path to the database file
    abs_db_path = os.path.abspath(db_path)
    duckdb_ui_logger.info(f"Opening DuckDB database: {abs_db_path}")

    # Show the database path to the user
    st.info(f"Opening DuckDB database: {abs_db_path}")

    # Check if the database file exists
    if not os.path.exists(abs_db_path):
        st.error(f"Database file not found: {abs_db_path}")
        duckdb_ui_logger.error(f"Database file not found: {abs_db_path}")
        return False

    # First, aggressively close all existing connections to the database
    duckdb_ui_logger.info(f"Forcing close of all connections to {db_path}")
    duckdb_manager.force_close_all_connections(db_path)

    # Launch DuckDB's built-in web interface if available
    try:
        import webbrowser

        import duckdb

        # Check DuckDB version
        duckdb_version = duckdb.__version__
        duckdb_ui_logger.info(f"DuckDB version: {duckdb_version}")

        # Connect to the database to check capabilities
        conn = duckdb.connect(abs_db_path, read_only=True)

        # Check if the GUI method exists (available in DuckDB 0.9.0+)
        if hasattr(conn, "gui") and callable(conn.gui):
            # Show a message about launching the UI
            st.info(
                "Launching DuckDB Web Interface. This will open in a new browser tab."
            )

            # Launch the GUI in a separate thread
            import threading

            def run_duckdb_web():
                try:
                    # Open browser first
                    webbrowser.open("http://localhost:8080")
                    # Then start the server
                    conn.gui(port=8080)
                except Exception as e:
                    duckdb_ui_logger.error(f"Error in DuckDB web thread: {e}")
                    st.error(f"Error launching DuckDB GUI: {e}")

            # Start the thread
            web_thread = threading.Thread(target=run_duckdb_web)
            web_thread.daemon = (
                True  # Make thread a daemon so it exits when the main program exits
            )
            web_thread.start()

            # Show success message
            st.success(
                "DuckDB Web Interface launched. Check your browser for a new tab."
            )
            duckdb_ui_logger.info("DuckDB Web Interface launched successfully")
            return True
        else:
            # GUI method not available, show alternative options
            conn.close()
            duckdb_ui_logger.warning(
                f"DuckDB GUI method not available in version {duckdb_version}"
            )

            st.warning(
                f"The native DuckDB GUI is not available in your DuckDB version ({duckdb_version})."
            )
            st.info(
                "To use the native GUI feature, please upgrade to DuckDB 0.9.0 or later with: `pip install --upgrade duckdb`"
            )

            # Show alternative access methods
            st.subheader("Alternative ways to access this database:")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Option 1: Use DBeaver**")
                st.markdown("""
                DBeaver is a free universal database tool that works well with DuckDB.

                1. Download [DBeaver](https://dbeaver.io/download/)
                2. Create a new DuckDB connection
                3. Browse to the database file location shown above
                """)

            with col2:
                st.markdown("**Option 2: Use Python**")
                st.markdown(f"""
                ```python
                import duckdb
                conn = duckdb.connect('{abs_db_path}')
                result = conn.execute("SELECT * FROM companies LIMIT 5").fetchall()
                print(result)
                ```
                """)

            return False
    except Exception as e:
        duckdb_ui_logger.error(f"Error launching DuckDB Web Interface: {e}")
        st.error(f"Error launching DuckDB Web Interface: {e}")
        return False
