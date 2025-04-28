#!/bin/bash
echo "Checking dependencies..."
poetry run python examples/install_dependencies.py
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Please install them manually."
    echo "Required packages: python-dotenv streamlit plotly pandas duckdb"
    read -p "Press Enter to continue..."
    exit 1
fi

echo "Starting SEC Filing Analyzer Demo..."
poetry run streamlit run examples/finance_streamlit_demo.py
