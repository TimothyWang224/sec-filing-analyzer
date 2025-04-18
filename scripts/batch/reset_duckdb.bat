@echo off
echo Resetting DuckDB database...
cd %~dp0\..\..
python -m scripts.reset_duckdb
echo Done!
pause
