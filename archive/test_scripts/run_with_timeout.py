"""
Run Streamlit with Timeout

This script runs Streamlit with a timeout to prevent it from hanging.
"""

import os
import subprocess
import sys
import time


def run_with_timeout(command, timeout_seconds=10):
    """Run a command with a timeout."""
    print(f"Running command: {command}")
    print(f"Timeout: {timeout_seconds} seconds")

    # Start the process
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)

    # Wait for the process to complete or timeout
    start_time = time.time()
    while process.poll() is None:
        # Check if we've exceeded the timeout
        if time.time() - start_time > timeout_seconds:
            print(f"Process timed out after {timeout_seconds} seconds")
            process.terminate()
            break

        # Sleep to avoid using 100% CPU
        time.sleep(0.1)

    # Get the output
    stdout, stderr = process.communicate()

    # Print the output
    print("\nStandard Output:")
    print(stdout)

    print("\nStandard Error:")
    print(stderr)

    # Return the return code
    return process.returncode


def main():
    """Main function."""
    # Get the command from the command line
    if len(sys.argv) < 2:
        print("Usage: python run_with_timeout.py <command>")
        sys.exit(1)

    command = " ".join(sys.argv[1:])

    # Run the command with a timeout
    return_code = run_with_timeout(command)

    print(f"\nProcess exited with return code: {return_code}")


if __name__ == "__main__":
    main()
