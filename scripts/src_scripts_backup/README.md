# Source Scripts Backup

This directory contains scripts that were previously located in the `src/scripts` directory. These scripts have been moved here as part of the directory structure consolidation.

## Purpose

These scripts are primarily for development, testing, and exploration purposes. They include:

1. Database exploration and testing scripts (e.g., `explore_duckdb.py`, `test_duckdb_store.py`)
2. XBRL extraction and testing scripts (e.g., `test_edgar_xbrl_extractor.py`, `extract_xbrl_data.py`)
3. ETL pipeline testing scripts (e.g., `run_nvda_etl.py`, `run_multi_company_etl.py`)
4. Vector store testing scripts (e.g., `test_faiss_persistence.py`, `test_optimized_vector_store.py`)
5. Miscellaneous utility scripts (e.g., `analyze_token_usage.py`, `cleanup_databases.py`)

## Usage

Most of these scripts are not meant to be used directly in production. They are kept here for reference and potential reuse in future development.

If you need to use any of these scripts, consider moving them to the appropriate subdirectory in the `scripts` directory or refactoring them into proper modules in the `src/sec_filing_analyzer` package.

## Key ETL Scripts

### `run_nvda_etl.py`

Process SEC filings for NVIDIA Corporation (NVDA).

```bash
python -m scripts.src_scripts_backup.run_nvda_etl NVDA --start-date 2023-01-01 --end-date 2023-12-31 --filing-types 10-K 10-Q
```

### `run_multi_company_etl.py`

Process SEC filings for multiple companies.

```bash
python -m scripts.src_scripts_backup.run_multi_company_etl --tickers AAPL MSFT NVDA --start-date 2023-01-01 --end-date 2023-12-31
```

Note: The paths in the examples above have been updated to reflect the new location of these scripts.
