#!/usr/bin/env python
"""
Pre-commit wrapper script that logs all output from pre-commit hooks.
This script is called by pre-commit as a hook itself.
"""

import datetime
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    # Create logs directory if it doesn't exist
    # Use a directory inside the git repository but excluded by .gitignore
    logs_dir = Path(".logs") / "precommit"
    logs_dir.mkdir(exist_ok=True, parents=True)

    # Generate timestamp and log file paths
    # Use ISO format with timezone information for the log header
    now = datetime.datetime.now().astimezone()
    iso_timestamp = now.isoformat(timespec="seconds")

    # Use a filename-friendly format for the file name
    file_timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"precommit_{file_timestamp}.log"
    latest_log = logs_dir / "latest.log"

    # Run pre-commit and capture output
    try:
        # Open the log file
        with open(log_file, "w") as log:
            # Write header information
            log.write(f"Pre-commit run at {iso_timestamp}\n")
            log.write(f"Working directory: {os.getcwd()}\n")
            log.write("-" * 80 + "\n\n")

            # Run pre-commit with all hooks except our wrapper
            # Only run on staged files, not all files
            cmd = ["pre-commit", "run", "--hook-stage", "pre-commit"]
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "SKIP": "log-wrapper"},  # Skip our wrapper to avoid infinite recursion
            )

            # Process the output to add hook-specific headers
            output = process.stdout

            # Add headers for each hook
            hook_pattern = re.compile(r"^([a-zA-Z0-9_-]+)\.+([A-Za-z]+)$", re.MULTILINE)
            processed_output = hook_pattern.sub(r"\n===== \1 (\2) =====\n\1\2", output)

            # Write the processed output to the log file
            log.write(processed_output)

            # Also print the original output to the console
            print(output, end="")

            # Write footer
            log.write("\n" + "-" * 80 + "\n")
            log.write(f"Exit code: {process.returncode}\n")

        # Create a copy as latest.log (using shutil.copy to ensure proper file rotation on all platforms)
        shutil.copy(log_file, latest_log)

        # Return the original process return code
        # This ensures that if other hooks fail, pre-commit will still fail
        # But we'll exit with 0 in the main function to avoid the log wrapper itself being reported as failed
        return process.returncode
    except Exception as e:
        print(f"Error in pre-commit wrapper: {e}")
        # Return error code if our wrapper fails
        return 1


if __name__ == "__main__":
    # Run the main function to execute pre-commit hooks and log the output
    result = main()

    # Always exit with code 0 to prevent pre-commit from thinking the log wrapper itself failed
    # The actual pre-commit hooks will still block commits if they fail through the normal pre-commit mechanism
    sys.exit(0)
