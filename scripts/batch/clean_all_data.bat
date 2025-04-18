@echo off
echo WARNING: This will delete all data from all storage systems!
echo.

:: First run in dry-run mode to see what would be deleted
echo Running in dry-run mode first...
poetry run python scripts\clean_all_data.py --dry-run

echo.
echo.
echo If you want to proceed with actual deletion, press any key.
echo To cancel, press Ctrl+C.
pause

:: Run the actual deletion
echo.
echo Running actual deletion...
poetry run python scripts\clean_all_data.py

:: Keep the window open
pause
