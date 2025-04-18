@echo off
echo Running Data Explorer...

:: Run Streamlit directly
poetry run streamlit run src\streamlit_app\pages\data_explorer.py

:: Keep the window open
pause
