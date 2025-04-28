#!/usr/bin/env python
"""
Pytest wrapper script that logs all output from pytest runs.
This script can be called directly or through a pre-commit hook.
"""

import datetime
import os
import subprocess
import sys
import shutil
from pathlib import Path


def main():
    # Get the repository root (assuming the script is in the scripts directory)
    repo_root = Path(__file__).parent.parent

    # Create logs directory if it doesn't exist
    # Use a directory inside the git repository but excluded by .gitignore
    logs_dir = repo_root / ".logs" / "pytest"
    logs_dir.mkdir(exist_ok=True, parents=True)

    # Print the log directory for debugging
    print(f"Pytest logs will be saved to: {logs_dir.absolute()}")

    # Generate timestamp and log file paths
    # Use ISO format with timezone information for the log header
    now = datetime.datetime.now().astimezone()
    iso_timestamp = now.isoformat(timespec="seconds")

    # Use a filename-friendly format for the file name
    file_timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pytest_{file_timestamp}.log"
    xml_file = logs_dir / f"pytest_{file_timestamp}.xml"
    latest_log = logs_dir / "latest.log"
    latest_xml = logs_dir / "latest.xml"

    # Build pytest command with arguments
    pytest_args = sys.argv[1:] if len(sys.argv) > 1 else []

    # Always add the JUnit XML output
    cmd = ["pytest", f"--junitxml={xml_file}"]

    # Add any additional arguments passed to this script
    cmd.extend(pytest_args)

    # Run pytest and capture output
    try:
        # Open the log file
        with open(log_file, "w") as log:
            # Write header information
            log.write(f"Pytest run at {iso_timestamp}\n")
            log.write(f"Script directory: {Path(__file__).parent.absolute()}\n")
            log.write(f"Working directory: {os.getcwd()}\n")
            log.write(f"Repository root: {repo_root.absolute()}\n")
            log.write(f"Command: {' '.join(cmd)}\n")
            log.write("-" * 80 + "\n\n")

            # Run pytest and capture output
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Write the output to the log file
            log.write(process.stdout)

            # Also print the original output to the console
            print(process.stdout, end="")

            # Write footer
            log.write("\n" + "-" * 80 + "\n")
            log.write(f"Exit code: {process.returncode}\n")

        # Create copies as latest files (using shutil.copy to ensure proper file rotation on all platforms)
        shutil.copy(log_file, latest_log)
        if os.path.exists(xml_file):
            shutil.copy(xml_file, latest_xml)

        # Return the original exit code to properly report test failures
        return process.returncode
    except Exception as e:
        print(f"Error in pytest wrapper: {e}")
        # Return error code if our wrapper fails
        return 1


if __name__ == "__main__":
    sys.exit(main())
