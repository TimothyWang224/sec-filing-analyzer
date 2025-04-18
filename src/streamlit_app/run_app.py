"""
Run Streamlit App

This script launches the Streamlit application for the SEC Filing Analyzer.
"""

import os
import sys
import socket
import shutil
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import configuration
try:
    from sec_filing_analyzer.config import ConfigProvider, StreamlitConfig
except ImportError as e:
    print(f"Error importing configuration: {e}")
    print("Make sure the SEC Filing Analyzer package is installed correctly.")
    sys.exit(1)

def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

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

def check_streamlit_installed():
    """Check if Streamlit is installed."""
    return shutil.which("streamlit") is not None

def main():
    """Launch the Streamlit application."""
    # Check if Streamlit is installed
    if not check_streamlit_installed():
        print("Error: Streamlit is not installed or not in PATH.")
        print("Please install Streamlit using: pip install streamlit")
        sys.exit(1)

    # Initialize configuration
    print("Initializing configuration...")
    try:
        ConfigProvider.initialize()
        streamlit_config = ConfigProvider.get_config(StreamlitConfig)
    except Exception as e:
        print(f"Error initializing configuration: {e}")
        print("Using default configuration values.")
        # Use default values if configuration fails
        class DefaultConfig:
            port = 8501
            headless = True
            theme_base = "light"
            enable_cors = True
            enable_xsrf_protection = False
        streamlit_config = DefaultConfig()

    # Get the port
    initial_port = streamlit_config.port
    print(f"Configured port: {initial_port}")

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
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = str(streamlit_config.headless).lower()
    env["STREAMLIT_THEME_BASE"] = streamlit_config.theme_base
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_SERVER_ENABLE_CORS"] = str(getattr(streamlit_config, 'enable_cors', True)).lower()
    env["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = str(getattr(streamlit_config, 'enable_xsrf_protection', False)).lower()

    # Get the path to the app.py file
    app_path = Path(__file__).resolve().parent / "app.py"
    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)

    # Launch Streamlit
    print(f"Launching SEC Filing Analyzer Streamlit app on port {port}...")
    print(f"App path: {app_path}")
    print("If the browser doesn't open automatically, please visit:")
    print(f"http://localhost:{port}")

    # Run Streamlit directly (not in a subprocess)
    os.environ.update(env)
    os.system(f"streamlit run {app_path}")

if __name__ == "__main__":
    main()
