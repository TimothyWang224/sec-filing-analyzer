"""
DuckDB Explorer Script

This script provides a simple interface to explore the tables in a DuckDB database.
"""

import argparse

import duckdb
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()


def list_tables(conn):
    """List all tables in the database."""
    tables = conn.execute("SHOW TABLES").fetchall()

    if not tables:
        console.print("[yellow]No tables found in the database.[/yellow]")
        return

    table = Table(title="Tables in Database", box=box.ROUNDED)
    table.add_column("Table Name", style="cyan")

    for t in tables:
        table.add_row(t[0])

    console.print(table)


def describe_table(conn, table_name):
    """Describe a table's schema."""
    try:
        schema = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()

        if not schema:
            console.print(f"[yellow]Table '{table_name}' exists but has no columns.[/yellow]")
            return

        table = Table(title=f"Schema for '{table_name}'", box=box.ROUNDED)
        table.add_column("Column", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Nullable", style="yellow")
        table.add_column("Default", style="blue")
        table.add_column("Primary Key", style="red")

        for col in schema:
            table.add_row(
                col[1],  # name
                col[2],  # type
                "YES" if col[3] == 0 else "NO",  # notnull (0 = nullable)
                str(col[4]) if col[4] is not None else "",  # dflt_value
                "YES" if col[5] == 1 else "NO",  # pk
            )

        console.print(table)

        # Show row count
        count = conn.execute(f"SELECT COUNT(*) FROM '{table_name}'").fetchone()[0]
        console.print(f"[bold]Row count:[/bold] {count}")

    except duckdb.Error as e:
        console.print(f"[red]Error describing table '{table_name}': {e}[/red]")


def show_sample(conn, table_name, limit=10):
    """Show a sample of rows from the table."""
    try:
        # Get column names first
        columns = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        column_names = [col[1] for col in columns]

        # Fetch sample data
        sample = conn.execute(f"SELECT * FROM '{table_name}' LIMIT {limit}").fetchall()

        if not sample:
            console.print(f"[yellow]Table '{table_name}' is empty.[/yellow]")
            return

        # Create a rich table
        table = Table(title=f"Sample Data from '{table_name}' (First {limit} rows)", box=box.ROUNDED)

        # Add columns
        for col in column_names:
            table.add_column(col, overflow="fold")

        # Add rows
        for row in sample:
            table.add_row(*[str(val) if val is not None else "NULL" for val in row])

        console.print(table)

    except duckdb.Error as e:
        console.print(f"[red]Error showing sample from table '{table_name}': {e}[/red]")


def run_query(conn, query):
    """Run a custom SQL query."""
    try:
        # Execute the query
        result = conn.execute(query).fetchall()

        if not result:
            console.print("[yellow]Query returned no results.[/yellow]")
            return

        # Get column names from the query
        column_names = [desc[0] for desc in conn.description]

        # Create a rich table
        table = Table(title="Query Results", box=box.ROUNDED)

        # Add columns
        for col in column_names:
            table.add_column(col, overflow="fold")

        # Add rows
        for row in result:
            table.add_row(*[str(val) if val is not None else "NULL" for val in row])

        console.print(table)

    except duckdb.Error as e:
        console.print(f"[red]Error executing query: {e}[/red]")


def show_table_relationships(conn):
    """Show foreign key relationships between tables."""
    try:
        # Get all tables
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]

        relationships = []

        # For each table, check for foreign keys
        for table_name in table_names:
            try:
                # This is a bit of a hack since DuckDB doesn't have a direct way to get foreign keys
                # We'll look for columns that end with "_id" or have "foreign key" in their comments
                schema = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()

                for col in schema:
                    col_name = col[1]
                    if col_name.endswith("_id") or "ticker" in col_name.lower():
                        # This is likely a foreign key
                        # Try to guess the referenced table
                        ref_table = col_name.replace("_id", "")
                        if ref_table in table_names or col_name in ["ticker", "filing_id", "fact_id"]:
                            if col_name == "ticker":
                                ref_table = "companies"
                            elif col_name == "filing_id":
                                ref_table = "filings"
                            elif col_name == "fact_id":
                                ref_table = "financial_facts"

                            relationships.append((table_name, col_name, ref_table))
            except:
                pass

        if not relationships:
            console.print("[yellow]No relationships found between tables.[/yellow]")
            return

        table = Table(title="Table Relationships", box=box.ROUNDED)
        table.add_column("Table", style="cyan")
        table.add_column("Foreign Key", style="green")
        table.add_column("Referenced Table", style="yellow")

        for rel in relationships:
            table.add_row(rel[0], rel[1], rel[2])

        console.print(table)

    except duckdb.Error as e:
        console.print(f"[red]Error showing table relationships: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Explore a DuckDB database")
    parser.add_argument("--db", default="data/financial_data.duckdb", help="Path to the DuckDB database file")
    parser.add_argument("--list-tables", action="store_true", help="List all tables in the database")
    parser.add_argument("--describe", help="Describe a specific table")
    parser.add_argument("--sample", help="Show a sample of rows from a table")
    parser.add_argument("--limit", type=int, default=10, help="Limit for sample rows (default: 10)")
    parser.add_argument("--query", help="Run a custom SQL query")
    parser.add_argument("--relationships", action="store_true", help="Show table relationships")

    args = parser.parse_args()

    try:
        # Connect to the database
        conn = duckdb.connect(args.db)
        console.print(Panel(f"Connected to DuckDB database: [bold cyan]{args.db}[/bold cyan]"))

        # Execute the requested command
        if args.list_tables:
            list_tables(conn)
        elif args.describe:
            describe_table(conn, args.describe)
        elif args.sample:
            show_sample(conn, args.sample, args.limit)
        elif args.query:
            run_query(conn, args.query)
        elif args.relationships:
            show_table_relationships(conn)
        else:
            # If no specific command is given, show a menu
            console.print("\n[bold]Available commands:[/bold]")
            console.print("  1. List tables")
            console.print("  2. Describe a table")
            console.print("  3. Show sample data")
            console.print("  4. Run a custom query")
            console.print("  5. Show table relationships")
            console.print("  0. Exit")

            choice = input("\nEnter your choice (0-5): ")

            if choice == "1":
                list_tables(conn)
            elif choice == "2":
                table_name = input("Enter table name: ")
                describe_table(conn, table_name)
            elif choice == "3":
                table_name = input("Enter table name: ")
                limit = input("Enter row limit (default: 10): ")
                limit = int(limit) if limit.isdigit() else 10
                show_sample(conn, table_name, limit)
            elif choice == "4":
                query = input("Enter SQL query: ")
                run_query(conn, query)
            elif choice == "5":
                show_table_relationships(conn)
            elif choice == "0":
                console.print("[bold green]Goodbye![/bold green]")
            else:
                console.print("[red]Invalid choice[/red]")

    except duckdb.Error as e:
        console.print(f"[red]Error connecting to database: {e}[/red]")
    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    main()
