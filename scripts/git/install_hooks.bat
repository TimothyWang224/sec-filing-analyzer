@echo off
echo Installing Git hooks...

REM Create the git directory if it doesn't exist
if not exist ".git\hooks" mkdir ".git\hooks"

REM Copy the post-commit hook
copy /Y "scripts\git\post-commit.bat" ".git\hooks\post-commit"
echo Post-commit hook installed.

echo.
echo Git hooks installed successfully.
echo Now your changes will be automatically pushed to GitHub after each commit.
pause
