"""
Organize scripts into a more robust directory structure with revised categories.

This script:
1. Creates a new directory structure for scripts
2. Moves scripts to their appropriate directories based on function, not naming
3. Creates README.md files for each directory
"""

import os
import shutil
from pathlib import Path

from rich.console import Console

console = Console()


def create_directory_structure():
    """Create the directory structure for scripts."""
    # Define the directories
    directories = [
        "scripts/etl",
        "scripts/db/duckdb",
        "scripts/db/neo4j",
        "scripts/analysis",
        "scripts/utils",
        "scripts/visualization",
        "scripts/maintenance",
        "scripts/examples",
    ]

    # Create the directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Created directory: {directory}[/green]")


def create_readme_files():
    """Create README.md files for each directory."""
    readme_content = {
        "scripts": """# SEC Filing Analyzer Scripts

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
""",
        "scripts/etl": """# ETL Scripts

This directory contains scripts for extracting, transforming, and loading SEC filing data.

## Available Scripts

- `run_nvda_etl.py`: Process SEC filings for NVIDIA Corporation (NVDA)
- `run_multi_company_etl.py`: Process SEC filings for multiple companies
- `run_multi_company_etl_parallel.py`: Process SEC filings for multiple companies in parallel
- `update_etl_pipeline.py`: Update the ETL pipeline
- `extract_xbrl_data.py`: Extract XBRL data from SEC filings
- `extract_xbrl_direct.py`: Extract XBRL data directly from SEC filings
- `fetch_msft_filing.py`: Fetch Microsoft SEC filings
- `list_msft_filings.py`: List Microsoft SEC filings
- `reprocess_aapl_filing.py`: Reprocess Apple SEC filings
- `reprocess_zero_vector_filings.py`: Reprocess filings with zero vectors
""",
        "scripts/db": """# Database Scripts

This directory contains scripts for working with databases.

## Subdirectories

- `duckdb/`: DuckDB-specific scripts
- `neo4j/`: Neo4j-specific scripts
""",
        "scripts/db/duckdb": """# DuckDB Scripts

This directory contains scripts for working with DuckDB databases.

## Available Scripts

- `duckdb_cli.py`: Command-line interface for DuckDB
- `explore_duckdb.py`: Explore DuckDB databases
- `query_duckdb.py`: Query DuckDB databases
- `init_db.py`: Initialize DuckDB databases
- `migrate_duckdb_schema.py`: Migrate DuckDB schema
- `simple_schema.py`: Create a simple DuckDB schema
- `test_duckdb.py`: Test DuckDB connection and basic operations
- `test_duckdb_store.py`: Test DuckDB store functionality
- `example_improved_duckdb.py`: Example of improved DuckDB schema
- `compare_duckdb_schemas.py`: Compare DuckDB schemas
- `check_db_schema.py`: Check DuckDB schema
""",
        "scripts/db/neo4j": """# Neo4j Scripts

This directory contains scripts for working with Neo4j databases.

## Available Scripts

(No specific Neo4j scripts yet)
""",
        "scripts/analysis": """# Analysis Scripts

This directory contains scripts for analyzing SEC filing data.

## Available Scripts

- `analyze_aapl_chunks.py`: Analyze Apple filing chunks
- `analyze_storage_efficiency.py`: Analyze storage efficiency
- `analyze_token_usage.py`: Analyze token usage
- `query_financial_data.py`: Query financial data
- `direct_search.py`: Direct search of SEC filings
- `test_advanced_search.py`: Test advanced search functionality
- `test_coordinated_search.py`: Test coordinated search functionality
- `test_vector_store.py`: Test vector store functionality
- `test_optimized_vector_store.py`: Test optimized vector store
- `test_faiss_persistence.py`: Test FAISS persistence
- `test_delta_index.py`: Test delta index functionality
""",
        "scripts/utils": """# Utility Scripts

This directory contains utility scripts for the SEC Filing Analyzer project.

## Available Scripts

- `migrate_data_structure.py`: Migrate data structure
- `migrate_to_numpy_storage.py`: Migrate to NumPy storage
- `test_openai_api.py`: Test OpenAI API
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
""",
        "scripts/visualization": """# Visualization Scripts

This directory contains scripts for visualizing SEC filing data.

## Available Scripts

- `launch_duckdb_explorer.py`: Launch DuckDB Explorer
- `launch_duckdb_web.py`: Launch DuckDB Web Explorer
- `simple_duckdb_explorer.py`: Simple DuckDB Explorer
- `streamlit_duckdb_explorer.py`: Streamlit DuckDB Explorer
- `explore_vector_store.py`: Explore vector store
""",
        "scripts/maintenance": """# Maintenance Scripts

This directory contains scripts for maintaining the SEC Filing Analyzer project.

## Available Scripts

- `cleanup_databases.py`: Clean up databases
- `cleanup_root_directory.py`: Clean up root directory
- `find_zero_vector_filings.py`: Find filings with zero vectors
""",
        "scripts/examples": """# Example Scripts

This directory contains example scripts demonstrating usage of the SEC Filing Analyzer components.

## Available Scripts

- `test_with_mock_data.py`: Example using mock data
- `test_with_simplified_schema.py`: Example using simplified schema
- `test_improved_xbrl.py`: Example of improved XBRL extraction
- `test_improved_xbrl_extractor.py`: Example of improved XBRL extractor
- `test_improved_edgar_extractor.py`: Example of improved Edgar extractor
- `test_parallel_xbrl.py`: Example of parallel XBRL processing
- `test_reorganized_pipeline.py`: Example of reorganized pipeline
- `test_reorganized_structure.py`: Example of reorganized structure
- `test_simplified_xbrl.py`: Example of simplified XBRL extraction
- `hello_world.py`: Simple hello world example
""",
    }

    # Create the README.md files
    for directory, content in readme_content.items():
        readme_path = Path(directory) / "README.md"
        with open(readme_path, "w") as f:
            f.write(content)
        console.print(f"[blue]Created README.md in {directory}[/blue]")


def organize_scripts():
    """Organize scripts into the directory structure."""
    # Define the script categories
    script_categories = {
        "scripts/etl": [
            "run_nvda_etl.py",
            "run_multi_company_etl.py",
            "run_multi_company_etl_parallel.py",
            "update_etl_pipeline.py",
            "extract_xbrl_data.py",
            "extract_xbrl_direct.py",
            "fetch_msft_filing.py",
            "fetch_msft_filing_direct.py",
            "fetch_msft_filing_direct_with_auth.py",
            "fetch_msft_filing_with_auth.py",
            "fetch_msft_first_8k_2022.py",
            "list_msft_filings.py",
            "reprocess_aapl_filing.py",
            "reprocess_zero_vector_filings.py",
            "explore_edgar_xbrl.py",
        ],
        "scripts/db/duckdb": [
            "duckdb_cli.py",
            "explore_duckdb.py",
            "query_duckdb.py",
            "init_db.py",
            "migrate_duckdb_schema.py",
            "simple_schema.py",
            "test_duckdb.py",
            "test_duckdb_store.py",
            "example_improved_duckdb.py",
            "compare_duckdb_schemas.py",
            "check_db_schema.py",
        ],
        "scripts/analysis": [
            "analyze_aapl_chunks.py",
            "analyze_storage_efficiency.py",
            "analyze_token_usage.py",
            "query_financial_data.py",
            "direct_search.py",
            "test_advanced_search.py",
            "test_coordinated_search.py",
            "test_vector_store.py",
            "test_optimized_vector_store.py",
            "test_faiss_persistence.py",
            "test_delta_index.py",
            "find_zero_vector_filings.py",
        ],
        "scripts/utils": [
            "migrate_data_structure.py",
            "migrate_to_numpy_storage.py",
            "test_openai_api.py",
            "check_edgar_package.py",
            "test_edgar_utils.py",
            "test_sec_downloader.py",
            "test_sec_downloader_xbrl.py",
            "test_edgar_library.py",
            "test_edgar_xbrl_basic.py",
            "test_edgar_xbrl_detailed.py",
            "test_edgar_xbrl_extractor.py",
            "test_edgar_xbrl_simple.py",
            "test_edgar_xbrl_to_duckdb.py",
            "test_specific_filing.py",
            "test_xbrl_extraction.py",
            "test_xbrl_extractor.py",
        ],
        "scripts/visualization": [
            "launch_duckdb_explorer.py",
            "launch_duckdb_web.py",
            "simple_duckdb_explorer.py",
            "streamlit_duckdb_explorer.py",
            "explore_vector_store.py",
        ],
        "scripts/maintenance": ["cleanup_databases.py", "cleanup_root_directory.py", "organize_scripts_revised.py"],
        "scripts/examples": [
            "test_with_mock_data.py",
            "test_with_simplified_schema.py",
            "test_improved_xbrl.py",
            "test_improved_xbrl_extractor.py",
            "test_improved_edgar_extractor.py",
            "test_parallel_xbrl.py",
            "test_reorganized_pipeline.py",
            "test_reorganized_structure.py",
            "test_simplified_xbrl.py",
            "hello_world.py",
            "init_and_test_improved_extractor.py",
        ],
    }

    # Source directories
    source_dirs = ["scripts", "src/scripts"]

    # Move scripts to their categories
    for category, script_list in script_categories.items():
        for script in script_list:
            # Check if the script exists in any of the source directories
            for source_dir in source_dirs:
                source_path = Path(source_dir) / script
                if source_path.exists():
                    # Create the destination path
                    dest_path = Path(category) / script

                    # Copy the script to the destination
                    shutil.copy2(source_path, dest_path)
                    console.print(f"[green]Copied {script} to {category}[/green]")
                    break
            else:
                console.print(f"[yellow]Script not found: {script}[/yellow]")

    console.print("\n[bold green]Script organization complete![/bold green]")
    console.print("[bold]Scripts have been organized into the following structure:[/bold]")
    console.print("- scripts/etl/")
    console.print("- scripts/db/duckdb/")
    console.print("- scripts/db/neo4j/")
    console.print("- scripts/analysis/")
    console.print("- scripts/utils/")
    console.print("- scripts/visualization/")
    console.print("- scripts/maintenance/")
    console.print("- scripts/examples/")
    console.print(
        "\n[bold]Note:[/bold] The original scripts have been copied, not moved. You may want to clean up the original script directories after verifying the new structure."
    )


def main():
    """Main function."""
    console.print("[bold]Organizing scripts into a more robust directory structure...[/bold]")

    # Create the directory structure
    create_directory_structure()

    # Create README.md files
    create_readme_files()

    # Organize scripts
    organize_scripts()


if __name__ == "__main__":
    main()
