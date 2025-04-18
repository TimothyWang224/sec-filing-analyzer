@echo off
echo Running ETL Pipeline...

:: Run the ETL pipeline
poetry run python scripts\run_etl_pipeline.py

:: Keep the window open
pause
