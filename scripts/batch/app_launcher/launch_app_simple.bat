@echo off
echo Launching SEC Filing Analyzer Streamlit App...

:: Run the app directly in the current window
echo Starting Streamlit server...
cd %~dp0\..\..\..\
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
