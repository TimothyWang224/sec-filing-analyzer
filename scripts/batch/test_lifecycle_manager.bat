@echo off
echo Testing Data Lifecycle Manager...

:: Run the test script
poetry run python scripts\test_lifecycle_manager.py

:: Keep the window open
pause
