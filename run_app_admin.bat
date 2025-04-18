@echo off
echo Launching SEC Filing Analyzer Streamlit App (Admin Version)...

:: Store the current directory
set CURRENT_DIR=%CD%

:: Set title for easy identification
title SEC Filing Analyzer Server

echo.
echo =====================================================
echo SEC FILING ANALYZER SERVER
echo =====================================================
echo.
echo Press 'Q' at any time to quit the server cleanly.
echo.

:: Find an available port using Python
echo Finding available port...
for /f "tokens=*" %%a in ('python find_port.py') do set PORT=%%a
echo Using port: %PORT%

:: Start the Streamlit app in the browser
start "SEC Filing Analyzer Browser" http://localhost:%PORT%

:: Run Streamlit in the background with the selected port
start /b cmd /c "poetry run streamlit run src\streamlit_app\app.py --server.port=%PORT% > streamlit_output.log 2>&1"

:: Store the PID of the background process
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq streamlit.exe" /fo list ^| findstr "PID:"') do set STREAMLIT_PID=%%a
echo Streamlit server started with PID: %STREAMLIT_PID%
echo.

echo Server is running. Press 'Q' to quit...

:check_key
choice /c Q /n /t 1 /d Q > nul 2>&1
if errorlevel 2 goto check_key
if errorlevel 1 goto shutdown

:shutdown
echo.
echo Shutting down Streamlit server...
taskkill /F /PID %STREAMLIT_PID% > nul 2>&1
taskkill /F /IM streamlit.exe > nul 2>&1
echo Server stopped.
pause
