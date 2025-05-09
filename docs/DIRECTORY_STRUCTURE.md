# SEC Filing Analyzer Directory Structure

This document provides an overview of the directory structure of the SEC Filing Analyzer project.

## Root Directory

The root directory contains only essential files:

- **README.md**: Main project documentation
- **CONTRIBUTING.md**: Contribution guidelines
- **SECURITY.md**: Security policy
- **RUNNING.md**: Instructions for running the application
- **pyproject.toml**: Project configuration for Poetry
- **poetry.lock**: Dependency lock file
- **.gitignore**: Git ignore file
- **.pre-commit-config.yaml**: Pre-commit configuration
- **pytest.ini**: PyTest configuration
- **.env.example**: Example environment variables
- **run_app.py**, **run_app.bat**: Scripts to run the main application
- **run_chat_app.py**, **run_chat_app.bat**, **run_chat_app.sh**: Scripts to run the chat application
- **run_chat_app_alt.py**, **run_chat_app_alt.bat**: Alternative scripts to run the chat application

## src Directory

The `src` directory contains the main source code for the application:

- **sec_filing_analyzer/**: Main package for the SEC Filing Analyzer
  - **agents/**: Agent implementations
  - **data_processing/**: Data processing utilities
  - **data_retrieval/**: Data retrieval utilities
  - **embeddings/**: Embedding generation utilities
  - **graphrag/**: Graph RAG components
  - **llm/**: LLM integration components
  - **pipeline/**: ETL pipeline implementation
  - **quantitative/**: Quantitative data processing
  - **search/**: Search functionality
  - **semantic/**: Semantic data processing
  - **storage/**: Storage utilities
  - **tools/**: Tool implementations for agents
  - **utils/**: Utility functions
- **agents/**: Agent implementations and core functionality
- **capabilities/**: Agent capabilities like planning and time awareness
- **environments/**: Environment definitions
- **llm/**: LLM integration components
- **memory/**: Memory implementations
- **streamlit_app/**: Streamlit application
- **tools/**: Tool implementations
- **utils/**: Utility functions
- **workflows/**: Workflow definitions
- **contracts.py**: Contract classes for agent-tool interactions
- **errors.py**: Error definitions
- **api.py**: API definitions

## scripts Directory

The `scripts` directory contains utility scripts and tools:

- **demo/**: Demo scripts for showcasing the project's capabilities
  - **run_nvda_etl.py**: Run the ETL pipeline for NVIDIA (demo version)
  - **query_revenue.py**: Query revenue data for a company (demo version)
  - **sec_downloader_patch.py**: Patch for the SEC downloader to support synthetic data
  - **README.md**: Documentation for the demo scripts
- **utils/**: Utility scripts for checking data, monitoring logs, etc.
  - **check_db.py**: Check the database status
  - **check_financial_data.py**: Check financial data in the database
  - **monitor_logs.py**: Monitor log files
- **maintenance/**: Scripts for project maintenance
  - **organize_root_directory.py**: Organize the root directory
  - **cleanup_root_directory.py**: Clean up the root directory
  - **list_and_delete_files.py**: List and delete files
- **etl/**: ETL scripts for processing SEC filings
  - **run_etl_pipeline.py**: Run the ETL pipeline for multiple companies
  - **run_nvda_etl.py**: Run the ETL pipeline for NVIDIA
- **data/**: Scripts for data management and manipulation
  - **add_nvda.py**: Add NVIDIA data to the database
  - **reset_duckdb.py**: Reset the DuckDB database
- **analysis/**: Scripts for analyzing data
  - **explore_vector_store.py**: Explore the vector store
  - **direct_search.py**: Direct search using cosine similarity
- **visualization/**: Scripts for visualizing data
  - **launch_log_visualizer.py**: Launch the log visualizer
- **database/**: Database-related scripts
  - **launch_duckdb_ui.py**: Launch the DuckDB UI
- **edgar/**: SEC EDGAR-related scripts
  - **reprocess_aapl_filing.py**: Reprocess Apple filings
- **tools/**: Utility tools for various tasks
  - **duckdb_explorer.py**: A standalone tool for exploring DuckDB databases
- **tests/**: Test scripts for various components
  - **test_vector_store.py**: Test the vector store
  - **test_unified_config.py**: Test the unified configuration
- **src_scripts_backup/**: Scripts moved from src/scripts during directory consolidation
  - Contains various development, testing, and exploration scripts
  - See the README.md in this directory for more details

## tests Directory

The `tests` directory contains unit and integration tests:

- **conftest.py**: PyTest configuration and fixtures
- **test_agents/**: Tests for agent implementations
- **test_config/**: Tests for configuration classes and utilities
- **test_contracts/**: Tests for contract classes
- **test_data_processing/**: Tests for data processing utilities
- **test_data_retrieval/**: Tests for data retrieval utilities
- **test_pipeline/**: Tests for ETL pipeline implementation
- **test_storage/**: Tests for storage utilities
- **test_utils/**: Tests for utility functions

## data Directory

The `data` directory contains data files and databases:

- **db_backup/**: Database backup files
  - **financial_data.duckdb**: DuckDB database with financial data
- **filings/**: SEC filing data
- **vector_store/**: Vector embeddings and metadata
- **logs/**: Log files

## docs Directory

The `docs` directory contains documentation:

- **DIRECTORY_STRUCTURE.md**: This document
- **API.md**: API documentation
- **ARCHITECTURE.md**: Architecture documentation
- **DEVELOPMENT.md**: Development documentation

## archive Directory

The `archive` directory contains archived files that are no longer needed but kept for reference:

- **root_cleanup/**: Files moved from the root directory during cleanup
- **batch_files/**: Old batch files for running the application
- **db_backup/**: Old database backup files
- **pyproject_backups/**: Backups of pyproject.toml during dependency updates
- **scripts/**: Old scripts that are no longer used
- **test_scripts/**: Old test scripts that are no longer used
- **semantic/**: Old semantic processing code
