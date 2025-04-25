"""
Find Available Port

This script finds an available port for the Streamlit server.
"""

import socket
import sys


def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return None


if __name__ == "__main__":
    # Default port
    default_port = 8501

    # Check if default port is available
    if not is_port_in_use(default_port):
        print(default_port)
        sys.exit(0)

    # Find an available port
    port = find_available_port(default_port + 1, max_attempts=20)

    if port:
        print(port)
        sys.exit(0)
    else:
        # If no port is available, use the default port anyway
        print(default_port)
        sys.exit(1)
