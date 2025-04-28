# Maintenance Scripts

This directory contains scripts for maintaining the SEC Filing Analyzer project.

## Available Scripts

- `cleanup_databases.py`: Clean up databases by removing temporary tables and optimizing storage
- `cleanup_root_directory.py`: Clean up root directory by removing temporary files
- `organize_root_directory.py`: Organize the root directory by moving files to appropriate subdirectories
- `delete_problematic_file.py`: Delete problematic files that can't be deleted through normal means
- `list_and_delete_files.py`: List all files in the root directory and delete them by index
- `consolidate_dependencies.py`: Consolidate dependencies in pyproject.toml
- `fix_notebook.py`: Fix issues with Jupyter notebooks

## Usage

### Organizing the Root Directory

To organize the root directory:

```bash
python scripts/maintenance/organize_root_directory.py
```

This script moves files from the root directory to appropriate subdirectories based on their function.

### Listing and Deleting Files

To list all files in the root directory and delete them by index:

```bash
python scripts/maintenance/list_and_delete_files.py
```

This script lists all files in the root directory with indices and allows you to delete them by entering the index.

### Cleaning Up Databases

To clean up databases:

```bash
python scripts/maintenance/cleanup_databases.py
```

This script removes temporary tables and optimizes storage in the DuckDB database.
