@echo off
echo Launching Simplified SEC Filing Analyzer App...

:: Set default port
set PORT=8503

:: Run Streamlit directly (not in a new window)
echo Starting Streamlit server on port %PORT%...
echo The command window will stay open to show any error messages.
echo DO NOT CLOSE THIS WINDOW while using the app.

:: Run the simplified app directly
streamlit run simplified_app.py --server.port=%PORT%

:: Keep the window open even if there's an error
echo.
echo If you see error messages above, please take note of them.
pause
