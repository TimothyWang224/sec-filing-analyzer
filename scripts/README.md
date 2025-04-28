# SEC Filing Analyzer Scripts

This directory contains scripts for the SEC Filing Analyzer project, organized by function.

## Directory Structure

- `demo/`: Demo scripts for showcasing the project's capabilities
  - `run_nvda_etl.py`: Run the ETL pipeline for NVIDIA (demo version)
  - `query_revenue.py`: Query revenue data for a company (demo version)
- `etl/`: ETL (Extract, Transform, Load) scripts for processing SEC filings
  - `run_etl_pipeline.py`: Run the ETL pipeline for multiple companies
  - `master_etl.py`: Master ETL script with comprehensive options
- `data/`: Scripts for data management and manipulation
  - `add_nvda.py`: Add NVIDIA data to the database
  - `reset_duckdb.py`: Reset the DuckDB database
- `utils/`: Utility scripts for checking data, monitoring logs, etc.
  - `check_db.py`: Check the database status
  - `check_financial_data.py`: Check financial data in the database
  - `monitor_logs.py`: Monitor log files
- `maintenance/`: Scripts for project maintenance
  - `organize_root_directory.py`: Organize the root directory
  - `prune_logs.py`: Prune log files
- `analysis/`: Data analysis scripts
  - `explore_vector_store.py`: Explore the vector store
  - `direct_search.py`: Direct search using cosine similarity
- `visualization/`: Data visualization scripts
  - `launch_log_visualizer.py`: Launch the log visualizer
- `database/`: Database-related scripts
  - `launch_duckdb_ui.py`: Launch the DuckDB UI
- `edgar/`: SEC EDGAR-related scripts
  - `reprocess_aapl_filing.py`: Reprocess Apple filings
- `tools/`: Utility tools for various tasks
  - `duckdb_explorer.py`: A standalone tool for exploring DuckDB databases
- `tests/`: Test scripts for various components
  - `test_vector_store.py`: Test the vector store
  - `test_unified_config.py`: Test the unified configuration

## Key Scripts

### Demo Scripts

The demo scripts provide a quick way to showcase the project's capabilities:

```bash
# Run ETL for NVIDIA (downloads real SEC filings)
poetry run python scripts/demo/run_nvda_etl.py --ticker NVDA --years 2023

# For offline testing, use synthetic data
TEST_MODE=True poetry run python scripts/demo/run_nvda_etl.py --ticker NVDA --years 2023

# Query revenue data
poetry run python scripts/demo/query_revenue.py --ticker NVDA --year 2023
```

### ETL Scripts

The ETL scripts process SEC filings and store the data in the database:

```bash
# Run the ETL pipeline for multiple companies
poetry run python scripts/etl/run_etl_pipeline.py --tickers AAPL MSFT NVDA --start-date 2023-01-01 --end-date 2023-12-31

# Run the master ETL script with comprehensive options
poetry run python scripts/etl/master_etl.py --tickers AAPL MSFT NVDA --start-date 2023-01-01 --end-date 2023-12-31 --index-type hnsw
```

### Utility Scripts

The utility scripts provide various tools for working with the project:

```bash
# Check the database status
poetry run python scripts/utils/check_db.py

# Monitor log files
poetry run python scripts/utils/monitor_logs.py

# Launch the DuckDB UI
poetry run python scripts/database/launch_duckdb_ui.py
```

For more details on specific scripts, see the README.md files in each subdirectory.
