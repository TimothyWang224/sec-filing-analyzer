# PowerShell script to run pre-commit and save logs
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = ".logs/precommit_$timestamp.log"
$latestLogFile = ".logs/latest.log"

# Run pre-commit and capture output
pre-commit run --all-files *> $logFile

# Create a copy as latest.log for easy access
Copy-Item $logFile $latestLogFile

# Get the exit code from pre-commit
$exitCode = $LASTEXITCODE

# Output the log location
Write-Host "Pre-commit log saved to: $logFile"
Write-Host "Latest log also available at: $latestLogFile"

# Return the original exit code
exit $exitCode
