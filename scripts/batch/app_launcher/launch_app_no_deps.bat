@echo off
echo Launching SEC Filing Analyzer Streamlit App (No Dependencies)...

:: Run the app with command window staying open
echo Starting Streamlit server...
echo The command window will stay open to show any error messages.
echo DO NOT CLOSE THIS WINDOW while using the app.

:: The run_app_no_deps.py script now automatically finds an available port
:: so we don't need to check for port availability here

:: Run directly (not in a new window) so we can see all output
cd %~dp0
python src\streamlit_app\run_app_no_deps.py

:: Keep the window open even if there's an error
echo.
echo If you see error messages above, please take note of them.
pause
