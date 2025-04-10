@echo off
echo Launching DuckDB Explorer on alternate port (8502)...

:: Check if the Streamlit server is already running on port 8502
netstat -ano | findstr :8502 > nul
if %errorlevel% equ 0 (
    echo Streamlit server is already running on port 8502.
    echo Opening browser...
    start http://localhost:8502
) else (
    echo Starting Streamlit server...
    start "DuckDB Explorer" cmd /c "poetry run streamlit run src/scripts/simple_duckdb_explorer.py --server.port=8502 && pause"
    
    :: Wait for the server to start
    echo Waiting for server to start...
    timeout /t 5 /nobreak > nul
    
    :: Open the browser
    echo Opening browser...
    start http://localhost:8502
)

echo.
echo If the browser doesn't open automatically, please visit:
echo http://localhost:8502
