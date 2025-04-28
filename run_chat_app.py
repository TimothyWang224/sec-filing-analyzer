"""
Run the SEC Filing Analyzer Chat App.

This script launches the Streamlit-based chat app for interacting with the
Financial Diligence Coordinator agent.
"""

import os
import socket
import subprocess
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
    """Run the SEC Filing Analyzer Chat App."""
    try:
        # Get the directory of this script
        script_dir = Path(__file__).resolve().parent

        # Set the working directory to the script directory
        os.chdir(script_dir)

        # Default port
        initial_port = 8501

        # Check if port is already in use
        if is_port_in_use(initial_port):
            print(f"Warning: Port {initial_port} is already in use.")
            print("Searching for an available port...")

            # Find an available port
            port = find_available_port(initial_port + 1, max_attempts=20)

            if port:
                print(f"Found available port: {port}")
            else:
                print(
                    "Could not find an available port. Using the configured port anyway."
                )
                port = initial_port
        else:
            port = initial_port
            print(f"Port {port} is available.")

        # Run the Streamlit app
        print(f"Starting SEC Filing Analyzer Chat App on port {port}...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "chat_app/app.py",
                f"--server.port={port}",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
    except KeyboardInterrupt:
        print("Chat app stopped by user.")


if __name__ == "__main__":
    main()
