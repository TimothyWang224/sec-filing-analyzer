@echo off
echo Running Enhanced ETL Data Inventory...

:: Kill any existing Streamlit processes
taskkill /F /IM streamlit.exe 2>nul

:: Run the sync storage script first
echo Synchronizing storage...
poetry run python scripts\sync_storage.py

:: Run Streamlit directly
echo Starting Streamlit app...
poetry run streamlit run src\streamlit_app\pages\etl_data_inventory.py

:: Keep the window open
pause
