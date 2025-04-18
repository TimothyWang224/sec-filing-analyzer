@echo off
echo Running Vector Store Explorer...

:: Run Streamlit directly
poetry run streamlit run src\streamlit_app\pages\vector_store_explorer.py

:: Keep the window open
pause
