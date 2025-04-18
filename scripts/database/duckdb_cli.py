"""
DuckDB CLI

A simple command-line interface for DuckDB.
"""

import argparse
import duckdb
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import os
import sys

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
                col[1],                                  # name
                col[2],                                  # type
                "YES" if col[3] == 0 else "NO",          # notnull (0 = nullable)
                str(col[4]) if col[4] is not None else "", # dflt_value
                "YES" if col[5] == 1 else "NO"           # pk
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

def interactive_mode(conn):
    """Interactive mode for the CLI."""
    console.print(Panel("DuckDB CLI Interactive Mode", style="bold green"))
    console.print("Type 'exit' or 'quit' to exit, 'help' for help.")
    
    while True:
        try:
            command = console.input("[bold cyan]duckdb>[/bold cyan] ").strip()
            
            if command.lower() in ['exit', 'quit']:
                break
            elif command.lower() == 'help':
                console.print(Panel("""
[bold]Available Commands:[/bold]

[cyan]tables[/cyan] - List all tables
[cyan]describe <table>[/cyan] - Describe a table's schema
[cyan]sample <table> [limit][/cyan] - Show a sample of rows from a table
[cyan]export <query> <filename>[/cyan] - Export query results to CSV
[cyan]<sql query>[/cyan] - Execute a SQL query

[bold]Examples:[/bold]

tables
describe companies
sample filings 5
SELECT * FROM companies LIMIT 5
export "SELECT * FROM companies" companies.csv
                """, title="Help", expand=False))
            elif command.lower() == 'tables':
                list_tables(conn)
            elif command.lower().startswith('describe '):
                table_name = command[9:].strip()
                describe_table(conn, table_name)
            elif command.lower().startswith('sample '):
                parts = command[7:].strip().split()
                table_name = parts[0]
                limit = int(parts[1]) if len(parts) > 1 else 10
                show_sample(conn, table_name, limit)
            elif command.lower().startswith('export '):
                parts = command[7:].strip().split(' ', 1)
                if len(parts) < 2:
                    console.print("[red]Error: export requires a query and filename[/red]")
                    continue
                
                query = parts[0]
                filename = parts[1]
                
                try:
                    df = conn.execute(query).fetchdf()
                    df.to_csv(filename, index=False)
                    console.print(f"[green]Exported {len(df)} rows to {filename}[/green]")
                except Exception as e:
                    console.print(f"[red]Error exporting data: {e}[/red]")
            else:
                run_query(conn, command)
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' or 'quit' to exit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description="DuckDB CLI")
    parser.add_argument("--db", default="data/financial_data.duckdb", help="Path to the DuckDB database file")
    parser.add_argument("--list-tables", action="store_true", help="List all tables in the database")
    parser.add_argument("--describe", help="Describe a specific table")
    parser.add_argument("--sample", help="Show a sample of rows from a table")
    parser.add_argument("--limit", type=int, default=10, help="Limit for sample rows (default: 10)")
    parser.add_argument("--query", help="Run a custom SQL query")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    try:
        # Connect to the database
        db_path = args.db
        if not os.path.exists(db_path):
            console.print(f"[red]Database file not found: {db_path}[/red]")
            return 1
        
        conn = duckdb.connect(db_path)
        console.print(Panel(f"Connected to DuckDB database: [bold cyan]{db_path}[/bold cyan]"))
        
        # Execute the requested command
        if args.list_tables:
            list_tables(conn)
        elif args.describe:
            describe_table(conn, args.describe)
        elif args.sample:
            show_sample(conn, args.sample, args.limit)
        elif args.query:
            run_query(conn, args.query)
        elif args.interactive:
            interactive_mode(conn)
        else:
            # If no specific command is given, start interactive mode
            interactive_mode(conn)
        
        return 0
        
    except duckdb.Error as e:
        console.print(f"[red]Error connecting to database: {e}[/red]")
        return 1
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    sys.exit(main())
