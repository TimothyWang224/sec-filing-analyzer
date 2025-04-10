@echo off
echo Launching Simple DuckDB Explorer...

:: Check if the Streamlit server is already running
netstat -ano | findstr :8501 > nul
if %errorlevel% equ 0 (
    echo Streamlit server is already running on port 8501.
    echo Opening browser...
    start http://localhost:8501
) else (
    echo Starting Streamlit server...
    start "Simple DuckDB Explorer" cmd /c "poetry run streamlit run src/scripts/simple_duckdb_explorer.py && pause"
    
    :: Wait for the server to start
    echo Waiting for server to start...
    timeout /t 5 /nobreak > nul
    
    :: Open the browser
    echo Opening browser...
    start http://localhost:8501
)

echo.
echo If the browser doesn't open automatically, please visit:
echo http://localhost:8501
