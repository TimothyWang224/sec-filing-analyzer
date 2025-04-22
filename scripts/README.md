# SEC Filing Analyzer Scripts

This directory contains scripts for the SEC Filing Analyzer project, organized by function.

## Directory Structure

- `etl/`: ETL (Extract, Transform, Load) scripts for processing SEC filings
- `db/`: Database-related scripts
  - `duckdb/`: DuckDB-specific scripts
  - `neo4j/`: Neo4j-specific scripts
- `analysis/`: Data analysis scripts
- `utils/`: Utility scripts
- `visualization/`: Data visualization scripts
- `maintenance/`: Maintenance and cleanup scripts
- `examples/`: Example scripts demonstrating usage
- `tools/`: Utility tools for various tasks
  - `duckdb_explorer.py`: A standalone tool for exploring DuckDB databases
- `tests/`: Test scripts for various components
  - `test_sync_mismatches.py`: Test script for the mismatch detection and fixing functionality
  - Various other test scripts for different components

## Recently Added Scripts

### DuckDB Explorer

The DuckDB Explorer is a standalone tool for exploring DuckDB databases. To run it:

```bash
python scripts/tools/duckdb_explorer.py
```

This will launch a Streamlit app that allows you to browse and query DuckDB databases.

### Mismatch Detection and Fixing

The `test_sync_mismatches.py` script tests the mismatch detection and fixing functionality. To run it:

```bash
python scripts/tests/test_sync_mismatches.py
```

This will detect and optionally fix mismatches between the database and file system.
