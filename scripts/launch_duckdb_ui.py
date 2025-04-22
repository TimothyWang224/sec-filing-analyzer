#!/usr/bin/env python
"""
Launch DuckDB UI

This script launches the DuckDB UI in a web browser.
It uses the DuckDB Web UI functionality to provide a web-based interface for exploring DuckDB databases.
"""

import argparse
import logging
import os
import sys
import webbrowser
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("launch_duckdb_ui")

def main():
    """Main function to launch DuckDB UI."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch DuckDB UI")
    parser.add_argument("--db", required=True, help="Path to DuckDB database file")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the DuckDB UI on")
    args = parser.parse_args()

    # Get absolute path to the database file
    db_path = os.path.abspath(args.db)
    logger.info(f"Database path: {db_path}")

    # Check if the database file exists
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        print(f"Error: Database file not found: {db_path}")
        return 1

    try:
        # Import DuckDB
        import duckdb
        logger.info(f"DuckDB version: {duckdb.__version__}")

        # Try to connect to the database
        conn = duckdb.connect(db_path, read_only=True)
        logger.info("Successfully connected to database")

        # Check if the database has tables
        tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
        table_names = [t[0] for t in tables]
        logger.info(f"Database contains {len(table_names)} tables: {', '.join(table_names)}")

        # Close the connection
        conn.close()
        logger.info("Database connection closed")

        # Launch the DuckDB Web UI
        try:
            # Check if duckdb_web_ui module is available
            try:
                import duckdb_web_ui
                logger.info("DuckDB Web UI module found")
            except ImportError:
                logger.error("DuckDB Web UI module not found. Please install it with: pip install duckdb-web-ui")
                print("Error: DuckDB Web UI module not found. Please install it with: pip install duckdb-web-ui")
                return 1

            # Launch the web UI
            port = args.port
            logger.info(f"Launching DuckDB Web UI on port {port}")
            
            # Open the browser
            webbrowser.open(f"http://localhost:{port}")
            
            # Start the server
            duckdb_web_ui.run_server(db_path, port=port)
            
            return 0
        except Exception as e:
            logger.error(f"Error launching DuckDB Web UI: {e}")
            print(f"Error launching DuckDB Web UI: {e}")
            
            # Fallback to alternative methods
            print("\nAlternative methods to access the database:")
            print(f"1. Use DBeaver: Connect to {db_path}")
            print(f"2. Use Python: import duckdb; conn = duckdb.connect('{db_path}')")
            print(f"3. Use DuckDB CLI: duckdb {db_path}")
            
            return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
