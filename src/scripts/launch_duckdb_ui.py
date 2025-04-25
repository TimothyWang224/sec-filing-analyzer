"""
Launch DuckDB UI

This script launches the DuckDB UI for a specified database file.
"""

import argparse
import logging
import os
import socket
import time
import webbrowser

import duckdb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("duckdb_ui_launcher")


def check_port_open(port, host="127.0.0.1", timeout=2):
    """Check if a port is open on the specified host."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


def launch_duckdb_ui(db_path, port=4213, listen="127.0.0.1"):
    """
    Launch the DuckDB UI for the specified database file.

    Args:
        db_path: Path to the DuckDB database file
        port: Port to use for the UI server
        listen: Interface to listen on

    Returns:
        True if successful, False otherwise
    """
    # Get absolute path to the database file
    abs_db_path = os.path.abspath(db_path)
    logger.info(f"Opening DuckDB database: {abs_db_path}")

    # Check if the database file exists
    if not os.path.exists(abs_db_path):
        logger.error(f"Database file not found: {abs_db_path}")
        return False

    try:
        # Create a new connection to the database
        logger.info(f"Creating new connection to {abs_db_path}")
        conn = duckdb.connect(abs_db_path, read_only=True)

        # Check if the connection is valid
        try:
            version = conn.execute("SELECT version()").fetchone()[0]
            logger.info(f"Successfully connected to DuckDB version: {version}")
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

        # Install and load the UI extension
        logger.info("Installing and loading UI extension")
        try:
            conn.execute("INSTALL ui")
            conn.execute("LOAD ui")
            logger.info("UI extension loaded successfully")
        except Exception as e:
            logger.warning(f"Error installing/loading UI extension: {e}")
            # Continue anyway, as it might already be installed

        # Start the UI server
        logger.info(f"Starting UI server on port {port}")
        try:
            conn.execute(f"CALL start_ui(port={port}, listen='{listen}', open_browser=false)")
            logger.info("UI server started successfully")

            # Wait a moment for the UI to start
            logger.info("Waiting for UI server to initialize")
            time.sleep(3)

            # Check if the server is running
            if check_port_open(port):
                logger.info(f"Port {port} is open, server is running")

                # Open the browser
                url = f"http://localhost:{port}"
                logger.info(f"Opening browser at {url}")
                webbrowser.open(url)

                # Keep the connection open to maintain the server
                logger.info("Server is running. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Stopping server...")
                    conn.close()
                    logger.info("Server stopped.")

                return True
            else:
                logger.error(f"Port {port} is not open, server failed to start")
                return False
        except Exception as e:
            logger.error(f"Error starting UI server: {e}")
            return False
    except Exception as e:
        logger.error(f"Error launching DuckDB UI: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch DuckDB UI")
    parser.add_argument(
        "--db", default="data/db_backup/improved_financial_data.duckdb", help="Path to the DuckDB database file"
    )
    parser.add_argument("--port", type=int, default=4213, help="Port to use for the UI server")
    parser.add_argument("--listen", default="127.0.0.1", help="Interface to listen on")

    args = parser.parse_args()

    # Launch the UI
    launch_duckdb_ui(args.db, args.port, args.listen)


if __name__ == "__main__":
    main()
