"""
Check DuckDB Issues

This script checks for potential issues with the DuckDB database.
"""

import os
import sys
import traceback
from pathlib import Path


def check_duckdb_database(db_path):
    """Check for potential issues with the DuckDB database."""
    print(f"Checking DuckDB database: {db_path}")

    # Check if the file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return False

    # Check file size
    file_size = os.path.getsize(db_path) / (1024 * 1024)  # Size in MB
    print(f"Database file size: {file_size:.2f} MB")

    # Check file permissions
    try:
        with open(db_path, "rb") as f:
            f.read(1)  # Try to read one byte
        print("Database file is readable")
    except Exception as e:
        print(f"Error reading database file: {e}")
        return False

    # Try to import duckdb
    try:
        import duckdb

        print("Successfully imported duckdb")
        print(f"DuckDB version: {duckdb.__version__}")
    except ImportError as e:
        print(f"Error importing duckdb: {e}")
        return False

    # Try to connect to the database
    try:
        conn = duckdb.connect(db_path)
        print("Successfully connected to database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print(traceback.format_exc())
        return False

    # Try to get the list of tables
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        print(f"Found {len(table_names)} tables: {table_names}")
    except Exception as e:
        print(f"Error getting tables: {e}")
        print(traceback.format_exc())
        return False

    # Check each table
    for table in table_names:
        print(f"\nChecking table: {table}")

        # Try to get the schema
        try:
            schema = conn.execute(f"DESCRIBE {table}").fetchall()
            print(f"Table schema: {schema}")
        except Exception as e:
            print(f"Error getting schema for table {table}: {e}")
            print(traceback.format_exc())
            continue

        # Try to get the row count
        try:
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"Table {table} has {row_count} rows")
        except Exception as e:
            print(f"Error getting row count for table {table}: {e}")
            print(traceback.format_exc())
            continue

        # Try to get a sample of data
        try:
            sample = conn.execute(f"SELECT * FROM {table} LIMIT 1").fetchall()
            print(f"Sample data from table {table}: {sample}")
        except Exception as e:
            print(f"Error getting sample data from table {table}: {e}")
            print(traceback.format_exc())
            continue

        # Check for problematic data types
        try:
            # Get column names and types
            columns = []
            for col in schema:
                columns.append((col[0], col[1]))

            # Check for BLOB columns
            blob_columns = [col[0] for col in columns if "BLOB" in col[1].upper()]
            if blob_columns:
                print(f"Warning: Table {table} has BLOB columns: {blob_columns}")

            # Check for JSON columns
            json_columns = [col[0] for col in columns if "JSON" in col[1].upper()]
            if json_columns:
                print(f"Warning: Table {table} has JSON columns: {json_columns}")

            # Check for very large TEXT columns
            for col_name, col_type in columns:
                if "TEXT" in col_type.upper() or "VARCHAR" in col_type.upper():
                    try:
                        max_length = conn.execute(f"SELECT MAX(LENGTH({col_name})) FROM {table}").fetchone()[0]
                        if max_length and max_length > 10000:
                            print(
                                f"Warning: Column {col_name} in table {table} has very large text values (max length: {max_length})"
                            )
                    except Exception as e:
                        print(f"Error checking length of column {col_name} in table {table}: {e}")
        except Exception as e:
            print(f"Error checking data types for table {table}: {e}")
            print(traceback.format_exc())

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
