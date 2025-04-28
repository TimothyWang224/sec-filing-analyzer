#!/usr/bin/env python
"""
Pre-commit log wrapper script.

This script wraps pre-commit hooks and logs their output to a file.
It's used to capture the output of pre-commit hooks for debugging purposes.
"""

import datetime
import os
import subprocess
import sys
from pathlib import Path

# Define log directory
LOG_DIR = Path(".logs/precommit")


def ensure_log_dir():
    """Ensure the log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_log_filename():
    """Generate a log filename based on the current timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"pre-commit_{timestamp}.log"


def run_pre_commit():
    """Run pre-commit and capture its output."""
    # Skip the log-wrapper hook to avoid infinite recursion
    os.environ["SKIP"] = "log-wrapper"

    try:
        # Create log file
        ensure_log_dir()
        log_file = get_log_filename()

        # Create a symbolic link to the latest log
        latest_link = LOG_DIR / "latest.log"
        if latest_link.exists():
            try:
                latest_link.unlink()
            except Exception as e:
                print(f"Warning: Could not remove latest.log link: {e}")

        try:
            # Create the latest.log link
            if sys.platform == "win32":
                # Windows doesn't support symbolic links easily, so just copy the file later
                pass
            else:
                latest_link.symlink_to(log_file.name)
        except Exception as e:
            print(f"Warning: Could not create latest.log link: {e}")

        # Open log file
        with open(log_file, "w", encoding="utf-8") as f:
            # Write header
            f.write(f"Pre-commit log - {datetime.datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            # Run pre-commit with all hooks except our wrapper
            # Only run on staged files, not all files
            cmd = ["pre-commit", "run", "--hook-stage", "pre-commit"]

            # Log the command
            f.write(f"Running command: {' '.join(cmd)}\n\n")

            # Run the command and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            # Stream output to both console and log file
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    sys.stdout.write(output)
                    f.write(output)
                    f.flush()

            # Wait for process to complete
            process.wait()

            # Write footer
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Exit code: {process.returncode}\n")

            # On Windows, create the "latest.log" by copying
            if sys.platform == "win32" and not latest_link.exists():
                try:
                    import shutil

                    shutil.copy2(log_file, latest_link)
                except Exception as e:
                    print(f"Warning: Could not create latest.log copy: {e}")

            # Return the original process return code
            # This ensures that if other hooks fail, pre-commit will still fail
            # But we'll exit with 0 in the main function to avoid the log wrapper itself being reported as failed
            return process.returncode

    except Exception as e:
        print(f"Error in pre-commit log wrapper: {e}")
        return 1


def main():
    """Main entry point."""
    return run_pre_commit()


if __name__ == "__main__":
    # Run the main function to execute pre-commit hooks and log the output
    result = main()

    # Always exit with code 0 to prevent pre-commit from thinking the log wrapper itself failed
    # The actual pre-commit hooks will still block commits if they fail through the normal pre-commit mechanism
    sys.exit(0)
