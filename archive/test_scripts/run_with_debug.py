"""
Run Streamlit with Debug Mode

This script runs the Streamlit app with debug mode enabled to show more detailed error messages.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run Streamlit with debug mode."""
    # Set environment variables for debugging
    env = os.environ.copy()
    env["STREAMLIT_LOGGER_LEVEL"] = "debug"
    env["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is not buffered

    # Get the path to the app.py file
    app_path = Path("src") / "streamlit_app" / "app.py"

    if not app_path.exists():
        print(f"Error: App file not found at {app_path}")
        sys.exit(1)

    # Run Streamlit with debug mode
    print(f"Running Streamlit with debug mode: {app_path}")

    # Use subprocess.Popen to keep the process running and capture output
    process = subprocess.Popen(
        ["poetry", "run", "streamlit", "run", str(app_path), "--logger.level=debug"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
    )

    # Print output in real-time
    print("Streamlit server starting...")
    print("Output will appear below. Press Ctrl+C to stop.")
    print("-" * 50)

    try:
        # Print stdout in real-time
        for line in process.stdout:
            print(line, end="")

        # Print stderr in real-time
        for line in process.stderr:
            print(f"ERROR: {line}", end="")

    except KeyboardInterrupt:
        print("\nStopping Streamlit server...")
        process.terminate()

    # Wait for process to complete
    process.wait()

    print("-" * 50)
    print("Streamlit server stopped.")


if __name__ == "__main__":
    main()
