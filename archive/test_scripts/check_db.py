import duckdb


def check_database(db_path):
    try:
        conn = duckdb.connect(db_path)
        tables = conn.execute("SHOW TABLES").fetchall()

        print(f"\nDatabase: {db_path}")
        print(f"Tables: {len(tables)}")

        for table in tables:
            table_name = table[0]
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  - {table_name}: {count} rows")

            # Get schema
            schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
            print(f"    Schema: {len(schema)} columns")
            for col in schema[:5]:  # Show first 5 columns only
                print(f"      - {col[0]}: {col[1]}")
            if len(schema) > 5:
                print(f"      - ... and {len(schema) - 5} more columns")

        conn.close()
    except Exception as e:
        print(f"Error with {db_path}: {str(e)}")


if __name__ == "__main__":
    check_database("data/financial_data.duckdb")
    check_database("data/db_backup/improved_financial_data.duckdb")
