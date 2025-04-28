#!/usr/bin/env python
"""
SEC Filing Analyzer App Launcher

This script launches the SEC Filing Analyzer Streamlit app.
It works on all platforms (Windows, macOS, Linux).
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Launch the SEC Filing Analyzer Streamlit app."""
    print("Launching SEC Filing Analyzer Streamlit App...")

    # Get the path to the run_app.py script
    run_app_path = (
        Path(__file__).resolve().parent / "src" / "streamlit_app" / "run_app.py"
    )

    if not run_app_path.exists():
        print(f"Error: App launcher not found at {run_app_path}")
        print(
            "Please make sure you're running this script from the project root directory."
        )
        sys.exit(1)

    # Check if we're running in a Poetry environment
    in_poetry = "POETRY_ACTIVE" in os.environ

    try:
        if in_poetry:
            # We're already in a Poetry environment, just run Python
            cmd = [sys.executable, str(run_app_path)]
        else:
            # We need to use Poetry to run the script
            cmd = ["poetry", "run", "python", str(run_app_path)]

        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error launching app: {e}")
        sys.exit(1)
    except FileNotFoundError:
        if not in_poetry:
            print(
                "Error: Poetry not found. Please install Poetry or activate your virtual environment."
            )
        else:
            print("Error: Python not found. Please check your environment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
