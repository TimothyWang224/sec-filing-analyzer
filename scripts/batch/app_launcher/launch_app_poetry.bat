@echo off
echo Launching SEC Filing Analyzer Streamlit App with Poetry...

:: Run the app with Poetry
echo Starting Streamlit server...
echo The command window will stay open to show any error messages.
echo DO NOT CLOSE THIS WINDOW while using the app.

:: Run directly (not in a new window) so we can see all output
poetry run python src\streamlit_app\run_app.py

:: Keep the window open even if there's an error
echo.
echo If you see error messages above, please take note of them.
pause
