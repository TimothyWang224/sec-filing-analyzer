@echo off
setlocal enabledelayedexpansion

:: Create a log file
set LOGFILE=%TEMP%\sec_filing_analyzer_launch.log
echo SEC Filing Analyzer Launch Log > %LOGFILE%
echo Timestamp: %date% %time% >> %LOGFILE%
echo. >> %LOGFILE%

echo Launching SEC Filing Analyzer Streamlit App...
echo Launching SEC Filing Analyzer Streamlit App... >> %LOGFILE%

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH. >> %LOGFILE%
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python and make sure it's in your PATH.
    echo See log file for details: %LOGFILE%
    pause
    exit /b 1
)

:: Check if Poetry is installed
where poetry >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Poetry is not installed or not in PATH. >> %LOGFILE%
    echo ERROR: Poetry is not installed or not in PATH.
    echo Please install Poetry and make sure it's in your PATH.
    echo See log file for details: %LOGFILE%
    pause
    exit /b 1
)

:: Check if the run_app.py file exists
if not exist "src\streamlit_app\run_app.py" (
    echo ERROR: run_app.py not found at src\streamlit_app\run_app.py >> %LOGFILE%
    echo ERROR: run_app.py not found at src\streamlit_app\run_app.py
    echo Please make sure you're running this script from the project root directory.
    echo See log file for details: %LOGFILE%
    pause
    exit /b 1
)

:: Run the app with output redirected to log file
echo Starting Streamlit server...
echo Starting Streamlit server... >> %LOGFILE%
echo Command: poetry run python src\streamlit_app\run_app.py >> %LOGFILE%

:: Run the app in the current window with output to console and log file
echo Running the app...
poetry run python src\streamlit_app\run_app.py 2>&1 | tee -a %LOGFILE%

:: Check if the app started successfully
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to start the Streamlit app. >> %LOGFILE%
    echo ERROR: Failed to start the Streamlit app.
    echo See log file for details: %LOGFILE%
    pause
    exit /b 1
)

echo.
echo If the browser doesn't open automatically, please check the
echo output above for the correct URL to use.
echo.
echo Press any key to close this window...
pause > nul
