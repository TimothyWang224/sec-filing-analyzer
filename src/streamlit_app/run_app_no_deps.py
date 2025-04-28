"""
Run Streamlit App (No Dependencies)

This script launches the Streamlit application without requiring the sec_filing_analyzer package.
"""

import os
import socket
import sys
from pathlib import Path


def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port.

    Args:
        start_port: The port to start checking from
        max_attempts: Maximum number of ports to check

    Returns:
        An available port or None if no port is available
    """
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return None


def main():
    """Launch the Streamlit application."""
    # Default configuration
    initial_port = 8501
    headless = True
    theme_base = "light"
    enable_cors = True
    enable_xsrf_protection = False

    # Check if port is already in use
    if is_port_in_use(initial_port):
        print(f"Warning: Port {initial_port} is already in use.")
        print("Searching for an available port...")

        # Find an available port
        port = find_available_port(initial_port + 1, max_attempts=20)

        if port:
            print(f"Found available port: {port}")
        else:
            print("Could not find an available port. Using the configured port anyway.")
            port = initial_port
    else:
        port = initial_port
        print(f"Port {port} is available.")

    # Set environment variables for Streamlit
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = str(headless).lower()
    os.environ["STREAMLIT_THEME_BASE"] = theme_base
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = str(enable_cors).lower()
    os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = str(enable_xsrf_protection).lower()

    # Get the path to the app_no_deps.py file
    app_path = Path(__file__).resolve().parent / "app_no_deps.py"
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)

    # Launch Streamlit
    print(f"Launching SEC Filing Analyzer Streamlit app on port {port}...")
    print(f"App path: {app_path}")
    print("If the browser doesn't open automatically, please visit:")
    print(f"http://localhost:{port}")

    # Run Streamlit directly
    os.system(f"streamlit run {app_path}")


if __name__ == "__main__":
    main()
