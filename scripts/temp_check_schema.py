"""
Temporary script to check DuckDB schema
"""

import duckdb


def check_schema():
    """Check the schema of the DuckDB database."""
    try:
        # Connect to DuckDB
        conn = duckdb.connect("data/financial_data.duckdb")

        # Get list of tables
        tables = conn.execute("SHOW TABLES").fetchdf()
        print(f"Tables in database: {', '.join(tables['name'].tolist())}")

        # Describe each table
        for table in tables["name"]:
            print(f"\n--- {table} ---")
            schema = conn.execute(f"DESCRIBE {table}").fetchdf()
            print(schema)

            # Get sample data
            print(f"\nSample data from {table}:")
            sample = conn.execute(f"SELECT * FROM {table} LIMIT 5").fetchdf()
            print(sample)

            # Get row count
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"Total rows in {table}: {count}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_schema()
