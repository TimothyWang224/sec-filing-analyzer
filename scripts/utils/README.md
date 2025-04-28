# Utility Scripts

This directory contains utility scripts for the SEC Filing Analyzer project.

## Available Scripts

### Data Management
- `migrate_data_structure.py`: Migrate data structure to a new format
- `migrate_to_numpy_storage.py`: Migrate vector embeddings to NumPy storage format
- `check_db.py`: Check the database status and structure
- `check_financial_data.py`: Check financial data in the database
- `check_metrics_schema.py`: Check the metrics schema in the database
- `check_companies.py`: Check the companies in the database
- `check_nvda_data.py`: Check NVIDIA data in the database
- `check_googl_data.py`: Check Google data in the database

### Monitoring and Debugging
- `monitor_logs.py`: Monitor log files in real-time
- `debug_sec_financial_data.py`: Debug SEC financial data extraction
- `analyze_timing.py`: Analyze timing of various operations
- `convert_workflow_log.py`: Convert workflow logs to a different format
- `migrate_logs.py`: Migrate logs to a new format

### Testing
- `test_openai_api.py`: Test OpenAI API connectivity and functionality
- `check_edgar_package.py`: Test Edgar package functionality
- `test_edgar_utils.py`: Test Edgar utilities
- `test_sec_downloader.py`: Test SEC downloader
- `test_sec_downloader_xbrl.py`: Test SEC downloader for XBRL
- `test_edgar_library.py`: Test Edgar library functionality
- `test_edgar_xbrl_basic.py`: Test Edgar XBRL basic functionality
- `test_edgar_xbrl_detailed.py`: Test Edgar XBRL detailed functionality
- `test_edgar_xbrl_extractor.py`: Test Edgar XBRL extractor
- `test_edgar_xbrl_simple.py`: Test Edgar XBRL simple functionality
- `test_edgar_xbrl_to_duckdb.py`: Test Edgar XBRL to DuckDB
- `test_specific_filing.py`: Test specific filing functionality
- `test_xbrl_extraction.py`: Test XBRL extraction
- `test_xbrl_extractor.py`: Test XBRL extractor

### Utilities
- `stop_streamlit.bat`: Stop Streamlit server on Windows

## Usage

### Checking Database Status

To check the database status:

```bash
python scripts/utils/check_db.py
```

### Monitoring Logs

To monitor logs in real-time:

```bash
python scripts/utils/monitor_logs.py
```

### Testing OpenAI API

To test OpenAI API connectivity:

```bash
python scripts/utils/test_openai_api.py
```

### Stopping Streamlit Server

To stop the Streamlit server on Windows:

```bash
scripts\utils\stop_streamlit.bat
```
