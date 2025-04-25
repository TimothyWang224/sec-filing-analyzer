"""
Direct Launcher for Streamlit App

This script provides a direct way to launch the Streamlit application
without relying on batch files or shell scripts.
"""

import os
import sys
import time
import webbrowser
from pathlib import Path


def main():
    """Launch the Streamlit application directly."""
    print("Launching SEC Filing Analyzer Streamlit App...")

    # Get the path to the run_app.py file
    app_script_path = Path(__file__).resolve().parent / "src" / "streamlit_app" / "run_app.py"

    if not app_script_path.exists():
        print(f"Error: App script not found at {app_script_path}")
        sys.exit(1)

    # Set the default port
    default_port = 8501

    # Launch the browser after a delay
    def open_browser():
        time.sleep(3)  # Wait for Streamlit to start
        webbrowser.open(f"http://localhost:{default_port}")

    # Import threading to open browser in background
    import threading

    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    # Print instructions
    print(f"The app will be available at: http://localhost:{default_port}")
    print("If the browser doesn't open automatically, please visit the URL manually.")

    # Run the Streamlit app
    print("Starting Streamlit server...")
    os.system(f"{sys.executable} {app_script_path}")


if __name__ == "__main__":
    main()
