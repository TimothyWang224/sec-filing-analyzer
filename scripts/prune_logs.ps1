# PowerShell script to prune old log files
# Keep only the 50 most recent log files

# Get all log files sorted by last write time (newest first)
$logDir = Join-Path -Path $HOME -ChildPath ".sec_filing_analyzer_logs"
$logFiles = Get-ChildItem -Path $logDir -Filter "precommit_*.log" | Sort-Object LastWriteTime -Descending

# If we have more than 50 files, delete the oldest ones
if ($logFiles.Count -gt 50) {
    $filesToDelete = $logFiles | Select-Object -Skip 50

    foreach ($file in $filesToDelete) {
        Remove-Item $file.FullName -Force
        Write-Host "Deleted old log file: $($file.Name)"
    }

    Write-Host "Pruned $($filesToDelete.Count) old log files. Keeping the 50 most recent."
} else {
    Write-Host "Found $($logFiles.Count) log files. No pruning needed (threshold is 50)."
}
