#!/bin/bash
# Bash script to run pre-commit and save logs

# Create logs directory if it doesn't exist
mkdir -p .logs

# Generate timestamp and log file paths
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file=".logs/precommit_${timestamp}.log"
latest_log=".logs/latest.log"

# Run pre-commit and capture output
pre-commit run --all-files 2>&1 | tee "$log_file"
exit_code=${PIPESTATUS[0]}

# Create a copy as latest.log for easy access
cp "$log_file" "$latest_log"

# Output the log location
echo "Pre-commit log saved to: $log_file"
echo "Latest log also available at: $latest_log"

# Return the original exit code
exit $exit_code
