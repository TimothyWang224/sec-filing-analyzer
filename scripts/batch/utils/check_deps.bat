@echo off
echo Checking dependencies...

:: Run the check dependencies script
poetry run python check_dependencies.py

:: Keep the window open
pause
