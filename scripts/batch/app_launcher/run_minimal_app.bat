@echo off
echo Launching Minimal Streamlit App...

:: Set default port
set PORT=8502

:: Run the minimal app
echo Starting Streamlit server on port %PORT%...
start "Minimal Streamlit App" cmd /c "streamlit run minimal_streamlit_app.py --server.port=%PORT% && pause"

:: Wait for the server to start
echo Waiting for server to start...
timeout /t 3 /nobreak > nul

:: Open the browser
echo Opening browser...
start http://localhost:%PORT%

echo.
echo If the browser doesn't open automatically, please visit:
echo http://localhost:%PORT%
echo.
echo If you encounter any issues, check the command window for error messages.
