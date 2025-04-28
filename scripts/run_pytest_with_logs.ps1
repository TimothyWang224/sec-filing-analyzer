# PowerShell script to run pytest with logging
# Usage: .\scripts\run_pytest_with_logs.ps1 [pytest arguments]

# Pass all arguments to the pytest wrapper script
python scripts/pytest_log_wrapper.py $args

# Return the exit code from pytest
exit $LASTEXITCODE
