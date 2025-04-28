"""
Clean up DuckDB databases.

This script:
1. Renames the best database (test_improved_edgar.duckdb) to financial_data_new.duckdb
2. Keeps test_mock_data.duckdb for reference
3. Deletes all other DuckDB databases
"""

import shutil
from pathlib import Path

from rich.console import Console

console = Console()


def cleanup_databases():
    """Clean up DuckDB databases."""
    # Define the data directory
    data_dir = Path("data")

    # Ensure the data directory exists
    if not data_dir.exists():
        console.print("[red]Data directory not found![/red]")
        return

    # List all DuckDB files
    duckdb_files = list(data_dir.glob("*.duckdb"))
    console.print(f"[bold]Found {len(duckdb_files)} DuckDB files[/bold]")

    # Check if the primary database exists
    primary_db = data_dir / "test_improved_edgar.duckdb"
    if not primary_db.exists():
        console.print("[red]Primary database (test_improved_edgar.duckdb) not found![/red]")
        return

    # Create a backup directory
    backup_dir = data_dir / "db_backup"
    backup_dir.mkdir(exist_ok=True)
    console.print(f"[green]Created backup directory: {backup_dir}[/green]")

    # Backup all databases
    for db_file in duckdb_files:
        backup_file = backup_dir / db_file.name
        shutil.copy2(db_file, backup_file)
        console.print(f"[blue]Backed up {db_file.name} to {backup_dir}[/blue]")

    # Rename the primary database
    new_primary_db = data_dir / "financial_data_new.duckdb"
    if new_primary_db.exists():
        # Backup the existing file first
        shutil.copy2(new_primary_db, backup_dir / new_primary_db.name)
        console.print(f"[yellow]Backed up existing {new_primary_db.name} to {backup_dir}[/yellow]")
        # Remove the existing file
        new_primary_db.unlink()
        console.print(f"[yellow]Removed existing {new_primary_db.name}[/yellow]")

    # Copy instead of rename to avoid issues with open files
    shutil.copy2(primary_db, new_primary_db)
    console.print(f"[green]Renamed {primary_db.name} to {new_primary_db.name}[/green]")

    # Keep the mock data database
    keep_db = data_dir / "test_mock_data.duckdb"
    console.print(f"[green]Keeping {keep_db.name} for reference[/green]")

    # Delete all other databases
    for db_file in duckdb_files:
        if db_file.name not in ["test_mock_data.duckdb", "test_improved_edgar.duckdb"]:
            try:
                db_file.unlink()
                console.print(f"[red]Deleted {db_file.name}[/red]")
            except Exception as e:
                console.print(f"[yellow]Could not delete {db_file.name}: {e}[/yellow]")

    console.print("\n[bold green]Database cleanup complete![/bold green]")
    console.print("[bold]Primary database: financial_data_new.duckdb[/bold]")
    console.print("[bold]Reference database: test_mock_data.duckdb[/bold]")
    console.print(f"[bold]Backups stored in: {backup_dir}[/bold]")


if __name__ == "__main__":
    cleanup_databases()
