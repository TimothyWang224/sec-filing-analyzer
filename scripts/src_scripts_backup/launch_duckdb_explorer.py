"""
Launch DuckDB Explorer

This script launches the Streamlit DuckDB Explorer.
"""

import argparse
import os
import subprocess
import time
import webbrowser


def main():
    parser = argparse.ArgumentParser(description="Launch the DuckDB Explorer")
    parser.add_argument(
        "--db",
        default="data/financial_data.duckdb",
        help="Path to the DuckDB database file",
    )

    args = parser.parse_args()

    # Ensure the database file exists
    db_path = args.db
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return

    # Set environment variables
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"  # Run in headless mode
    env["STREAMLIT_THEME_BASE"] = "light"  # Use light theme
    env["STREAMLIT_ONBOARDING_DISABLED"] = "true"  # Disable onboarding

    # Launch Streamlit
    streamlit_script = os.path.join(os.path.dirname(__file__), "streamlit_duckdb_explorer.py")
    cmd = ["streamlit", "run", streamlit_script, "--", "--db", db_path]

    print(f"Launching DuckDB Explorer for database: {db_path}")
    print("Starting Streamlit server...")

    # Start the Streamlit process
    process = subprocess.Popen(cmd, env=env)

    # Wait for the server to start
    time.sleep(3)

    # Open the browser
    url = "http://localhost:8501"
    print(f"Streamlit server started. Opening browser at {url}")
    print("Press Ctrl+C to stop the server.")

    # Try to open the browser
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Failed to open browser automatically: {e}")
        print(f"Please manually open {url} in your browser.")

    try:
        # Keep the script running
        process.wait()
    except KeyboardInterrupt:
        print("Stopping Streamlit server...")
        process.terminate()
        process.wait()
        print("Streamlit server stopped.")


if __name__ == "__main__":
    main()
