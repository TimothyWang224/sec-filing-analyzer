"""
DuckDB Query Script

This script allows you to run SQL queries against a DuckDB database.
"""

import argparse
import duckdb
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box

console = Console()

def run_query(conn, query, output_format="table", output_file=None):
    """Run a SQL query and display the results."""
    try:
        # Execute the query
        result = conn.execute(query).fetchall()
        
        if not result:
            console.print("[yellow]Query returned no results.[/yellow]")
            return
        
        # Get column names from the query
        column_names = [desc[0] for desc in conn.description]
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(result, columns=column_names)
        
        # Output based on format
        if output_format == "table":
            # Create a rich table
            table = Table(title="Query Results", box=box.ROUNDED)
            
            # Add columns
            for col in column_names:
                table.add_column(col, overflow="fold")
            
            # Add rows
            for row in result:
                table.add_row(*[str(val) if val is not None else "NULL" for val in row])
            
            console.print(table)
            
        elif output_format == "csv":
            if output_file:
                df.to_csv(output_file, index=False)
                console.print(f"[green]Results saved to {output_file}[/green]")
            else:
                console.print(df.to_csv(index=False))
                
        elif output_format == "json":
            if output_file:
                df.to_json(output_file, orient="records", lines=True)
                console.print(f"[green]Results saved to {output_file}[/green]")
            else:
                console.print(df.to_json(orient="records", indent=2))
                
        elif output_format == "pandas":
            console.print(df)
            
        # Return the DataFrame for further processing
        return df
        
    except duckdb.Error as e:
        console.print(f"[red]Error executing query: {e}[/red]")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run SQL queries against a DuckDB database")
    parser.add_argument("--db", default="data/financial_data.duckdb", help="Path to the DuckDB database file")
    parser.add_argument("--query", help="SQL query to run")
    parser.add_argument("--query-file", help="File containing SQL query to run")
    parser.add_argument("--format", choices=["table", "csv", "json", "pandas"], default="table", help="Output format")
    parser.add_argument("--output", help="Output file (for CSV or JSON format)")
    
    args = parser.parse_args()
    
    try:
        # Connect to the database
        conn = duckdb.connect(args.db)
        console.print(Panel(f"Connected to DuckDB database: [bold cyan]{args.db}[/bold cyan]"))
        
        # Get the query
        query = None
        if args.query:
            query = args.query
        elif args.query_file:
            with open(args.query_file, "r") as f:
                query = f.read()
        else:
            # Interactive mode
            console.print("[bold]Enter your SQL query (type 'exit' to quit):[/bold]")
            console.print("[dim]Example: SELECT * FROM companies LIMIT 5[/dim]")
            
            lines = []
            while True:
                line = input("> ")
                if line.lower() == "exit":
                    break
                lines.append(line)
                if line.strip().endswith(";"):
                    break
            
            query = "\n".join(lines)
        
        if query:
            # Run the query
            run_query(conn, query, args.format, args.output)
        
    except duckdb.Error as e:
        console.print(f"[red]Error connecting to database: {e}[/red]")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
