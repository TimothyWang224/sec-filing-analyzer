"""
Check the schema of the improved financial database.
"""

import duckdb
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def check_database_schema(db_path="data/db_backup/improved_financial_data.duckdb", read_only=True):
    """Check the schema of the database."""
    try:
        # Connect to DuckDB in read-only mode
        conn = duckdb.connect(db_path, read_only=read_only)
        print(f"Connected to database: {db_path}")

        # Get tables
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"Tables in database: {len(tables)}")

        for table in tables:
            table_name = table[0]
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"\n{table_name}: {count} rows")

            # Get schema
            schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
            print(f"  Schema: {len(schema)} columns")
            for col in schema:
                print(f"  - {col[0]}: {col[1]}")

            # Show sample data
            if count > 0:
                sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 1").fetchall()
                print(f"  Sample data: {sample}")

        # Close connection
        conn.close()
        print("\nDatabase schema check completed successfully")
        return True
    except Exception as e:
        print(f"Error checking database schema: {e}")
        return False

if __name__ == "__main__":
    check_database_schema()
