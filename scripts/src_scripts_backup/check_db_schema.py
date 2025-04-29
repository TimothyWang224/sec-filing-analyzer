"""
Check the schema of DuckDB databases.
"""

import os
import sys

import duckdb
from rich.console import Console
from rich.table import Table

console = Console()


def check_db_schema(db_path):
    """Check the schema of a DuckDB database."""
    if not os.path.exists(db_path):
        console.print(f"[red]Database file not found: {db_path}[/red]")
        return

    try:
        # Connect to the database
        conn = duckdb.connect(db_path)

        # Get all tables
        tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()

        console.print(f"[bold green]Database: {db_path}[/bold green]")
        console.print(f"[bold]Tables: {len(tables)}[/bold]")

        # Create a table for display
        table = Table(title="Tables")
        table.add_column("Table Name")
        table.add_column("Column Count")
        table.add_column("Row Count")

        for table_name in [t[0] for t in tables]:
            # Get column count
            columns = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            column_count = len(columns)

            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            table.add_row(table_name, str(column_count), str(row_count))

        console.print(table)

        # Check for specific improvements
        if any(t[0] == "companies" for t in tables):
            # Check if companies table has numeric ID
            company_schema = conn.execute("PRAGMA table_info('companies')").fetchall()
            company_id_column = next((col for col in company_schema if col[1] == "company_id"), None)

            if company_id_column:
                console.print(f"[green]✓ Companies table has numeric ID column (type: {company_id_column[2]})[/green]")
            else:
                console.print("[red]✗ Companies table does not have numeric ID column[/red]")

        # Check for metrics table (a key improvement)
        if any(t[0] == "metrics" for t in tables):
            console.print("[green]✓ Database has metrics table (improved schema)[/green]")
        else:
            console.print("[red]✗ Database does not have metrics table[/red]")

        # Check for facts table
        if any(t[0] == "facts" for t in tables):
            # Check facts table schema
            facts_schema = conn.execute("PRAGMA table_info('facts')").fetchall()
            facts_columns = [col[1] for col in facts_schema]

            # Check for normalized_value column (an improvement)
            if "normalized_value" in facts_columns:
                console.print("[green]✓ Facts table has normalized_value column (improved schema)[/green]")
            else:
                console.print("[red]✗ Facts table does not have normalized_value column[/red]")

        # Check for XBRL tag mappings table (another improvement)
        if any(t[0] == "xbrl_tag_mappings" for t in tables):
            console.print("[green]✓ Database has xbrl_tag_mappings table (improved schema)[/green]")
        else:
            console.print("[red]✗ Database does not have xbrl_tag_mappings table[/red]")

        # Close the connection
        conn.close()

    except Exception as e:
        console.print(f"[red]Error checking schema for {db_path}: {e}[/red]")


def main():
    # Get database paths from command line or use defaults
    if len(sys.argv) > 1:
        db_paths = sys.argv[1:]
    else:
        # Default database paths
        db_paths = [
            "data/financial_data.duckdb",
            "data/improved_financial_data.duckdb",
            "data/test_financial_data.duckdb",
            "data/test_improved_edgar.duckdb",
            "data/test_improved_financial_data.duckdb",
            "data/test_improved_xbrl.duckdb",
            "data/test_mock_data.duckdb",
            "data/test_simple_schema.duckdb",
        ]

    # Check each database
    for db_path in db_paths:
        check_db_schema(db_path)
        console.print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
