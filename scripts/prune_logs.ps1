# PowerShell script to prune old log files
# Keep only the 50 most recent log files

# Define the log directory
$logDir = Join-Path -Path $HOME -ChildPath ".sec_filing_analyzer_logs"

# Function to remove old logs of a specific type
function Remove-OldLogs {
    param (
        [string]$LogType,
        [string]$Pattern,
        [string]$SubDir = ""
    )

    $fullPath = if ($SubDir) { Join-Path -Path $logDir -ChildPath $SubDir } else { $logDir }

    # Create the directory if it doesn't exist
    if (-not (Test-Path $fullPath)) {
        New-Item -Path $fullPath -ItemType Directory -Force | Out-Null
        Write-Host "Created directory: $fullPath"
        return
    }

    # Get all log files sorted by last write time (newest first)
    $logFiles = Get-ChildItem -Path $fullPath -Filter $Pattern | Sort-Object LastWriteTime -Descending

    # If we have more than 50 files, delete the oldest ones
    if ($logFiles.Count -gt 50) {
        $filesToDelete = $logFiles | Select-Object -Skip 50

        foreach ($file in $filesToDelete) {
            Remove-Item $file.FullName -Force
            Write-Host "Deleted old $LogType log file: $($file.Name)"
        }

        Write-Host "Pruned $($filesToDelete.Count) old $LogType log files. Keeping the 50 most recent."
    } else {
        Write-Host "Found $($logFiles.Count) $LogType log files. No pruning needed (threshold is 50)."
    }
}

# Prune pre-commit logs
Remove-OldLogs -LogType "pre-commit" -Pattern "precommit_*.log"

# Prune pytest logs (both .log and .xml files)
Remove-OldLogs -LogType "pytest" -Pattern "pytest_*.log" -SubDir "pytest"
Remove-OldLogs -LogType "pytest" -Pattern "pytest_*.xml" -SubDir "pytest"
