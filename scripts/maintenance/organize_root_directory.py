#!/usr/bin/env python
"""
Organize the root directory of the SEC Filing Analyzer project.

This script moves files from the root directory to appropriate subdirectories
to maintain a clean and organized project structure.
"""

import os
import shutil
from pathlib import Path

# Define the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Define the destination directories
SCRIPTS_UTILS_DIR = ROOT_DIR / "scripts" / "utils"
SCRIPTS_DATA_DIR = ROOT_DIR / "scripts" / "data"
SCRIPTS_MAINTENANCE_DIR = ROOT_DIR / "scripts" / "maintenance"
SCRIPTS_DEMO_DIR = ROOT_DIR / "scripts" / "demo"
ARCHIVE_DIR = ROOT_DIR / "archive" / "root_cleanup"

# Create directories if they don't exist
for directory in [SCRIPTS_UTILS_DIR, SCRIPTS_DATA_DIR, SCRIPTS_MAINTENANCE_DIR, SCRIPTS_DEMO_DIR, ARCHIVE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Files to keep in the root directory
ROOT_FILES = [
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "pyproject.toml",
    "poetry.lock",
    ".gitignore",
    ".pre-commit-config.yaml",
    "pytest.ini",
    ".env.example",
    ".env",  # Keep .env in root but don't commit it
    "run_app.py",
    "run_app.bat",
    "run_chat_app.py",
    "run_chat_app.bat",
    "run_chat_app.sh",
    "run_chat_app_alt.py",
    "run_chat_app_alt.bat",
    "RUNNING.md",
    ".bandit.yaml",
    ".mypy.ini",
    ".pre-commit-log-wrapper.py",
]

# Files to move to scripts/utils/
UTILS_FILES = [
    "check_companies.py",
    "check_db.py",
    "check_financial_data.py",
    "check_googl_data.py",
    "check_metrics_schema.py",
    "check_nvda_data.py",
    "debug_sec_financial_data.py",
    "monitor_logs.py",
]

# Files to move to scripts/data/
DATA_FILES = [
    "add_nvda.py",
]

# Files to move to scripts/maintenance/
MAINTENANCE_FILES = [
    "fix_notebook.py",
]

# Files to delete
DELETE_FILES = [
    "data_management.log",
    "etl_data_inventory.log",
    "shutdown_signal.txt",
    "src.zip",
    "test_hook.txt",
    "rc.agents.coordinator import FinancialDiligenceCoordinator; coordinator = FinancialDiligenceCoordinator(); print('Successfully initialized coordinator with config-based specialized agents')",
]

# Files to archive
ARCHIVE_FILES = [
    "coverage.xml",
    ".coverage",
    "duckdb_explorer.html",
    "robust_embedding_test_results.json",
    "data_explorer_solution_guide.md",
    "README_CHAT_APP.md",
    "README_OPTIMIZED_PARALLEL_PROCESSING.md",
    "check_company_filings.py",
]


def move_files(files, destination):
    """Move files to the specified destination."""
    for file in files:
        source = ROOT_DIR / file
        if source.exists():
            dest = destination / file
            print(f"Moving {file} to {destination.relative_to(ROOT_DIR)}")
            shutil.move(str(source), str(dest))
        else:
            print(f"File not found: {file}")


def delete_files(files):
    """Delete files."""
    for file in files:
        source = ROOT_DIR / file
        if source.exists():
            print(f"Deleting {file}")
            os.remove(str(source))
        else:
            print(f"File not found: {file}")


def archive_files(files):
    """Archive files."""
    for file in files:
        source = ROOT_DIR / file
        if source.exists():
            dest = ARCHIVE_DIR / file
            print(f"Archiving {file}")
            shutil.move(str(source), str(dest))
        else:
            print(f"File not found: {file}")


def main():
    """Run the organization process."""
    print("Organizing root directory...")
    
    # Move files to scripts/utils/
    move_files(UTILS_FILES, SCRIPTS_UTILS_DIR)
    
    # Move files to scripts/data/
    move_files(DATA_FILES, SCRIPTS_DATA_DIR)
    
    # Move files to scripts/maintenance/
    move_files(MAINTENANCE_FILES, SCRIPTS_MAINTENANCE_DIR)
    
    # Delete files
    delete_files(DELETE_FILES)
    
    # Archive files
    archive_files(ARCHIVE_FILES)
    
    print("Root directory organization complete!")


if __name__ == "__main__":
    main()
