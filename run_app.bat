@echo off
echo Launching SEC Filing Analyzer Streamlit App...

:: Change to the project root directory (where this batch file is located)
cd %~dp0

:: Run the app using Poetry
echo Starting Streamlit server...
poetry run python src\streamlit_app\run_app.py

:: Keep the window open if there's an error
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Failed to start the Streamlit app.
    echo.
    pause
    exit /b 1
)

echo.
echo Press any key to close this window...
pause > nul
