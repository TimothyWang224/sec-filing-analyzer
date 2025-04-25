"""
Check DuckDB Database

This script checks if a DuckDB database is valid and can be opened.
"""

import os
import sys


def check_duckdb_database(db_path):
    """Check if a DuckDB database is valid and can be opened."""
    print(f"Checking DuckDB database: {db_path}")

    # Check if the file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return False

    # Try to import duckdb
    try:
        import duckdb

        print("Successfully imported duckdb")
    except ImportError as e:
        print(f"Error importing duckdb: {e}")
        return False

    # Try to connect to the database
    try:
        conn = duckdb.connect(db_path)
        print("Successfully connected to database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return False

    # Try to get the list of tables
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        print(f"Found {len(table_names)} tables: {table_names}")
    except Exception as e:
        print(f"Error getting tables: {e}")
        return False

    # Try to get the row count for each table
    for table in table_names:
        try:
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"Table {table} has {row_count} rows")
        except Exception as e:
            print(f"Error getting row count for table {table}: {e}")

    # Try to get a sample of data from each table
    for table in table_names:
        try:
            sample = conn.execute(f"SELECT * FROM {table} LIMIT 1").fetchdf()
            print(f"Sample data from table {table}:")
            print(sample)
        except Exception as e:
            print(f"Error getting sample data from table {table}: {e}")

    return True


def main():
    """Main function."""
    # Get the database path from the command line
    if len(sys.argv) < 2:
        db_path = "data/financial_data.duckdb"
        print(f"No database path provided, using default: {db_path}")
    else:
        db_path = sys.argv[1]

    # Check the database
    success = check_duckdb_database(db_path)

    if success:
        print("\nDatabase check completed successfully")
    else:
        print("\nDatabase check failed")


if __name__ == "__main__":
    main()
