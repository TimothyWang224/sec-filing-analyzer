#!/bin/bash
# Bash script to prune old log files
# Keep only the 50 most recent log files

# Create logs directory if it doesn't exist
LOG_DIR="$HOME/.sec_filing_analyzer_logs"
mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/pytest"

# Function to prune logs of a specific type
prune_logs() {
    local log_type="$1"
    local pattern="$2"
    local subdir="$3"

    # Set the full path
    local full_path="$LOG_DIR"
    if [ -n "$subdir" ]; then
        full_path="$LOG_DIR/$subdir"
        mkdir -p "$full_path"
    fi

    # Count the number of log files
    local log_count=$(ls -1 "$full_path"/$pattern 2>/dev/null | wc -l)

    if [ "$log_count" -gt 50 ]; then
        # Get the list of files to delete (all except the 50 newest)
        local files_to_delete=$(ls -1t "$full_path"/$pattern | tail -n +51)

        # Delete the files
        for file in $files_to_delete; do
            rm "$file"
            echo "Deleted old $log_type log file: $(basename "$file")"
        done

        # Calculate how many files were deleted
        local deleted_count=$((log_count - 50))
        echo "Pruned $deleted_count old $log_type log files. Keeping the 50 most recent."
    else
        echo "Found $log_count $log_type log files. No pruning needed (threshold is 50)."
    fi
}

# Prune pre-commit logs
prune_logs "pre-commit" "precommit_*.log" ""

# Prune pytest logs (both .log and .xml files)
prune_logs "pytest" "pytest_*.log" "pytest"
prune_logs "pytest" "pytest_*.xml" "pytest"
