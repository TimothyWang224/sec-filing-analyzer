#!/bin/bash
# Bash script to prune old log files
# Keep only the 50 most recent log files

# Create logs directory if it doesn't exist
LOG_DIR="$HOME/.sec_filing_analyzer_logs"
mkdir -p "$LOG_DIR"

# Count the number of log files
log_count=$(ls -1 "$LOG_DIR"/precommit_*.log 2>/dev/null | wc -l)

if [ "$log_count" -gt 50 ]; then
    # Get the list of files to delete (all except the 50 newest)
    files_to_delete=$(ls -1t "$LOG_DIR"/precommit_*.log | tail -n +51)

    # Delete the files
    for file in $files_to_delete; do
        rm "$file"
        echo "Deleted old log file: $(basename "$file")"
    done

    # Calculate how many files were deleted
    deleted_count=$((log_count - 50))
    echo "Pruned $deleted_count old log files. Keeping the 50 most recent."
else
    echo "Found $log_count log files. No pruning needed (threshold is 50)."
fi
