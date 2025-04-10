@echo off
echo Launching DuckDB Explorer...

:: Check if the Streamlit server is already running
netstat -ano | findstr :8501 > nul
if %errorlevel% equ 0 (
    echo Streamlit server is already running on port 8501.
    echo Opening browser...
    start http://localhost:8501
) else (
    echo Starting Streamlit server...
    start "DuckDB Explorer" cmd /c "poetry run python src/scripts/launch_duckdb_explorer.py %* && pause"

    :: Wait for the server to start
    echo Waiting for server to start...
    timeout /t 5 /nobreak > nul

    :: Open the browser
    echo Opening browser...
    start http://localhost:8501

    :: Open the HTML file as a fallback
    start duckdb_explorer.html
)

echo.
echo If the browser doesn't open automatically, please visit:
echo http://localhost:8501
