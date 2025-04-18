@echo off
echo Running Data Management Page...

:: Kill any existing Streamlit processes
taskkill /F /IM streamlit.exe 2>nul

:: Run Streamlit directly
echo Starting Streamlit app...
poetry run streamlit run src\streamlit_app\pages\data_management.py

:: Keep the window open
pause
