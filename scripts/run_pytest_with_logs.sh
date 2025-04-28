#!/bin/bash
# Shell script to run pytest with logging
# Usage: ./scripts/run_pytest_with_logs.sh [pytest arguments]

# Make the script executable if it isn't already
chmod +x scripts/pytest_log_wrapper.py

# Pass all arguments to the pytest wrapper script
python scripts/pytest_log_wrapper.py "$@"

# Return the exit code from pytest
exit $?
