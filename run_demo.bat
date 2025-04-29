@echo off
echo Checking dependencies...
poetry run python examples/install_dependencies.py
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies. Please install them manually.
    echo Required packages: python-dotenv streamlit plotly pandas duckdb
    pause
    exit /b 1
)

echo Starting SEC Filing Analyzer Demo...
poetry run streamlit run examples/finance_streamlit_demo.py --server.port 8502
