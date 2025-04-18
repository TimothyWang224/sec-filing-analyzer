@echo off
echo Launching SEC Filing Analyzer Streamlit App with Exit Button...

:: Delete any existing shutdown signal file
if exist shutdown_signal.txt del /f /q shutdown_signal.txt

:: Start the Streamlit app in a new process
start "SEC Filing Analyzer" cmd /c "poetry run streamlit run src\streamlit_app\app.py"

:: Wait for the server to start
echo Waiting for server to start...
timeout /t 3 /nobreak > nul

:: Open the browser
echo Opening browser...
start http://localhost:8501

:: Monitor for shutdown signal
echo Monitoring for shutdown signal...
echo The app will automatically close when you click the "Exit Application" button.
echo DO NOT close this window manually.

:check_shutdown
:: Check if shutdown signal file exists
if exist shutdown_signal.txt (
    echo Shutdown signal detected. Closing application...
    
    :: Kill the Streamlit process
    taskkill /F /IM streamlit.exe > nul 2>&1
    
    :: Delete the shutdown signal file
    del /f /q shutdown_signal.txt
    
    echo Application closed successfully.
    timeout /t 2 /nobreak > nul
    exit
) else (
    :: Wait for a moment before checking again
    timeout /t 2 /nobreak > nul
    goto check_shutdown
)
