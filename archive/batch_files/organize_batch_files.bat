@echo off
echo Organizing batch files...

:: Create directories
mkdir scripts\batch\data_explorer 2>nul
mkdir scripts\batch\duckdb_explorer 2>nul
mkdir scripts\batch\app_launcher 2>nul
mkdir scripts\batch\utils 2>nul

:: Move Data Explorer batch files
echo Moving Data Explorer batch files...
move run_fixed_data_explorer.bat scripts\batch\data_explorer\ 2>nul
move run_data_explorer.bat scripts\batch\data_explorer\ 2>nul
move run_data_explorer_simple.bat scripts\batch\data_explorer\ 2>nul
move run_explorer_lite.bat scripts\batch\data_explorer\ 2>nul
move run_simple_explorer.bat scripts\batch\data_explorer\ 2>nul
move run_minimal_explorer.bat scripts\batch\data_explorer\ 2>nul
move run_vector_store_explorer.bat scripts\batch\data_explorer\ 2>nul

:: Move DuckDB Explorer batch files
echo Moving DuckDB Explorer batch files...
move run_duckdb_explorer.bat scripts\batch\duckdb_explorer\ 2>nul
move run_duckdb_minimal.bat scripts\batch\duckdb_explorer\ 2>nul
move run_duckdb_connect.bat scripts\batch\duckdb_explorer\ 2>nul
move run_duckdb_tables.bat scripts\batch\duckdb_explorer\ 2>nul
move run_duckdb_explorer_alt.bat scripts\batch\duckdb_explorer\ 2>nul
move run_duckdb_explorer_debug.bat scripts\batch\duckdb_explorer\ 2>nul
move explore_duckdb.bat scripts\batch\duckdb_explorer\ 2>nul
move duckdb_cli.bat scripts\batch\duckdb_explorer\ 2>nul
move run_duckdb_cli.bat scripts\batch\duckdb_explorer\ 2>nul
move launch_duckdb_explorer.bat scripts\batch\duckdb_explorer\ 2>nul
move launch_simple_explorer.bat scripts\batch\duckdb_explorer\ 2>nul
move launch_explorer_alt_port.bat scripts\batch\duckdb_explorer\ 2>nul
move run_check_duckdb.bat scripts\batch\duckdb_explorer\ 2>nul

:: Move App Launch batch files
echo Moving App Launch batch files...
move launch_app.bat scripts\batch\app_launcher\ 2>nul
move launch_app_poetry.bat scripts\batch\app_launcher\ 2>nul
move launch_app_no_deps.bat scripts\batch\app_launcher\ 2>nul
move launch_streamlit_poetry.bat scripts\batch\app_launcher\ 2>nul
move launch_simplified_direct.bat scripts\batch\app_launcher\ 2>nul
move run_simplified_app.bat scripts\batch\app_launcher\ 2>nul
move run_minimal_app.bat scripts\batch\app_launcher\ 2>nul
move run_minimal_direct.bat scripts\batch\app_launcher\ 2>nul

:: Move Utility batch files
echo Moving Utility batch files...
move check_deps.bat scripts\batch\utils\ 2>nul

:: Create main launcher batch files in the root directory
echo Creating main launcher batch files...

:: Create app launcher
echo @echo off > run_app.bat
echo echo Launching SEC Filing Analyzer App... >> run_app.bat
echo. >> run_app.bat
echo :: Run the app launcher >> run_app.bat
echo call scripts\batch\app_launcher\launch_app.bat >> run_app.bat
echo. >> run_app.bat

:: Create data explorer launcher
echo @echo off > run_explorer.bat
echo echo Launching Data Explorer... >> run_explorer.bat
echo. >> run_explorer.bat
echo :: Run the data explorer >> run_explorer.bat
echo call scripts\batch\data_explorer\run_fixed_data_explorer.bat >> run_explorer.bat
echo. >> run_explorer.bat

:: Create DuckDB explorer launcher
echo @echo off > run_duckdb.bat
echo echo Launching DuckDB Explorer... >> run_duckdb.bat
echo. >> run_duckdb.bat
echo :: Run the DuckDB CLI >> run_duckdb.bat
echo call scripts\batch\duckdb_explorer\run_duckdb_cli.bat >> run_duckdb.bat
echo. >> run_duckdb.bat

echo Batch files organized successfully!
pause
