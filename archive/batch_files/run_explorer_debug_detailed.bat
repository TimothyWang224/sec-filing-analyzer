@echo off
echo Running Data Explorer with detailed logging...

:: Set environment variables for debugging
set STREAMLIT_LOGGER_LEVEL=debug
set PYTHONUNBUFFERED=1

:: Run Streamlit directly
echo Starting Streamlit server...
echo The command window will stay open to show any error messages.
echo DO NOT CLOSE THIS WINDOW while using the app.

:: Run directly (not in a new window) so we can see all output
poetry run streamlit run src\streamlit_app\pages\data_explorer.py --logger.level=debug

:: Keep the window open even if there's an error
echo.
echo If you see error messages above, please take note of them.
pause
