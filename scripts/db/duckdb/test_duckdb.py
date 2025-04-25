"""
Test if we can connect to DuckDB.
"""

import os

import duckdb


def main():
    print("Testing DuckDB connection...")

    # Create a test database
    db_path = "data/test_db.duckdb"

    # Make sure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    print(f"Connecting to {db_path}...")
    conn = duckdb.connect(db_path)

    print("Creating a test table...")
    conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER, name VARCHAR)")

    print("Inserting test data...")
    conn.execute("INSERT INTO test VALUES (1, 'test')")

    print("Querying test data...")
    result = conn.execute("SELECT * FROM test").fetchall()
    print(f"Result: {result}")

    print("Closing connection...")
    conn.close()

    print("Test complete.")


if __name__ == "__main__":
    main()
