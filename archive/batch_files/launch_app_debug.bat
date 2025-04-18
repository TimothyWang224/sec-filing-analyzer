@echo off
echo Launching SEC Filing Analyzer Streamlit App with debugging...

:: Set default port
set PORT=8501

:: Run the app with command window staying open
echo Starting Streamlit server on port %PORT%...
echo The command window will stay open to show any error messages.
echo DO NOT CLOSE THIS WINDOW while using the app.

:: Run directly (not in a new window) so we can see all output
cd %~dp0
python src\streamlit_app\run_app.py

:: Keep the window open even if there's an error
echo.
echo If you see error messages above, please take note of them.
pause
