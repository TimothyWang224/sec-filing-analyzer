#!/usr/bin/env python
"""
Launch Log Visualizer

A utility script to launch the SEC Filing Analyzer Log Visualizer Streamlit app.
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch SEC Filing Analyzer Log Visualizer")
    parser.add_argument("--log-file", help="Path to log file")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    args = parser.parse_args()

    # Find the workflow_visualizer.py script
    script_dir = Path(__file__).parent
    visualizer_script = script_dir / "workflow_visualizer.py"

    if not visualizer_script.exists():
        print(f"Error: Could not find workflow_visualizer.py in {script_dir}")
        sys.exit(1)

    # Set environment variables
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "true"  # Run in headless mode
    env["STREAMLIT_THEME_BASE"] = "light"  # Use light theme
    env["STREAMLIT_ONBOARDING_DISABLED"] = "true"  # Disable onboarding
    env["STREAMLIT_SERVER_PORT"] = str(args.port)  # Set port

    # Build command
    cmd = ["streamlit", "run", str(visualizer_script)]
    if args.log_file:
        cmd.extend(["--", "--log-file", args.log_file])

    print("Launching SEC Filing Analyzer Log Visualizer...")
    print(f"Command: {' '.join(cmd)}")

    # Start the Streamlit process
    process = subprocess.Popen(cmd, env=env)

    # Wait for the server to start
    time.sleep(3)

    # Open browser
    webbrowser.open(f"http://localhost:{args.port}")

    print(f"SEC Filing Analyzer Log Visualizer is running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop")

    try:
        # Wait for the process to complete
        process.wait()
    except KeyboardInterrupt:
        print("Stopping SEC Filing Analyzer Log Visualizer...")
        process.terminate()
        process.wait()


if __name__ == "__main__":
    main()
