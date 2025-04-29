# SEC Filing Analyzer Examples

This directory contains example scripts and applications that demonstrate how to use the SEC Filing Analyzer library.

## Available Examples

### Basic Usage

The `basic_usage.py` script demonstrates the core functionality of the SEC Filing Analyzer:

```bash
# Run the basic usage example
python examples/basic_usage.py
```

This example shows how to:
- Initialize the SEC Filing Analyzer
- Query financial data from SEC filings
- Perform semantic search on filing content
- Get insights from the financial analyst agent

### Chat Demo

The `run_chat_demo.py` script provides a simple command-line interface for chatting with the SEC Filing Analyzer:

```bash
# Run the chat demo
python -m examples.run_chat_demo
# Or use the Poetry script
poetry run chat-demo
```

This interactive demo allows you to:
- Ask questions about companies and their financial data
- Get insights from SEC filings
- Analyze financial metrics and trends

### Chat Streamlit Demo

The `streamlit_demo.py` script provides a web-based interface for interacting with the SEC Filing Analyzer:

```bash
# Run the Streamlit web demo
python -m examples.streamlit_demo
# Or use the Poetry script
poetry run chat-demo-web
```

This web demo provides:
- A user-friendly interface for querying SEC data
- Visualization of financial metrics
- Interactive chat with the SEC Filing Analyzer

### Finance Demo

The finance demo consists of several scripts that demonstrate the ETL pipeline and financial data querying capabilities:

#### ETL Pipeline Demo

The `run_nvda_etl.py` script demonstrates how to run the ETL pipeline for NVIDIA Corporation:

```bash
# Run the ETL pipeline for NVIDIA
python examples/run_nvda_etl.py --ticker NVDA --years 2023

# Use synthetic data for testing
python examples/run_nvda_etl.py --ticker NVDA --years 2023 --test-mode
```

This script:
- Downloads SEC filings for NVIDIA
- Extracts financial data from the filings
- Loads the data into the database
- Supports synthetic data mode for testing

#### Financial Data Query Demo

The `query_revenue.py` script demonstrates how to query financial data from the database:

```bash
# Query revenue data for NVIDIA in 2023
python examples/query_revenue.py --ticker NVDA --year 2023
```

This script:
- Connects to the DuckDB database
- Queries revenue data for a specific company and year
- Formats the output with citation information

#### Finance Streamlit Demo

The `finance_streamlit_demo.py` script provides a web-based interface for exploring financial data:

```bash
# Run the Finance Streamlit demo
python -m examples.finance_streamlit_demo
```

This web demo provides:
- A clean, modern UI for interacting with the SEC Filing Analyzer
- Interactive controls for running the ETL process
- Visualizations of financial data
- Quick revenue lookup functionality

## Running the Finance Demo

For a complete finance demo experience, you can use the provided batch files:

```bash
# On Windows
run_demo.bat

# On macOS/Linux
./run_demo.sh
```

These scripts will:
1. Check and install required dependencies
2. Launch the Finance Streamlit demo

## Adding New Examples

When adding new examples:

1. Place your example script in the `examples` directory
2. Add appropriate documentation in the script
3. Update this README with information about your example
4. Consider adding a Poetry script entry in `pyproject.toml` for easy execution
