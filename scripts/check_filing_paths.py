"""
Script to check filing paths in DuckDB
"""

import duckdb
import pandas as pd


def check_filing_paths():
    """Check if filing paths are stored in DuckDB."""
    try:
        # Connect to DuckDB
        conn = duckdb.connect("data/financial_data.duckdb")

        # Get filings data
        filings_df = conn.execute("SELECT * FROM filings").fetchdf()
        print("Filings table columns:")
        print(filings_df.columns.tolist())

        # Check if there's a document_url column and what it contains
        if "document_url" in filings_df.columns:
            print("\nDocument URLs:")
            for idx, row in filings_df.iterrows():
                print(f"{row['ticker']} - {row['filing_type']} - {row['document_url']}")

        # Check if there are any other tables that might store file paths
        print("\nChecking for file path columns in all tables:")
        tables = conn.execute("SHOW TABLES").fetchdf()
        for table in tables["name"]:
            schema = conn.execute(f"DESCRIBE {table}").fetchdf()
            path_columns = [
                col
                for col in schema["column_name"]
                if "path" in col.lower() or "url" in col.lower() or "file" in col.lower()
            ]
            if path_columns:
                print(f"Table {table} has potential path columns: {path_columns}")
                for col in path_columns:
                    sample = conn.execute(f"SELECT DISTINCT {col} FROM {table} LIMIT 5").fetchdf()
                    print(f"Sample values for {col}:")
                    print(sample)

        # Try to find any columns that might contain file paths
        print("\nSearching for potential file path values:")
        for table in tables["name"]:
            for col in conn.execute(f"DESCRIBE {table}").fetchdf()["column_name"]:
                if conn.execute(f"SELECT typeof({col}) FROM {table} LIMIT 1").fetchone()[0] == "VARCHAR":
                    # Check if any values look like file paths
                    path_check = conn.execute(f"""
                        SELECT {col} FROM {table} 
                        WHERE {col} IS NOT NULL 
                            AND ({col} LIKE '%.html' 
                                OR {col} LIKE '%.xml' 
                                OR {col} LIKE '%.txt'
                                OR {col} LIKE 'file:%'
                                OR {col} LIKE '/data/%'
                                OR {col} LIKE 'data/%'
                                OR {col} LIKE 'http%')
                        LIMIT 5
                    """).fetchdf()
                    if not path_check.empty:
                        print(f"Table {table}, Column {col} has potential file paths:")
                        print(path_check)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_filing_paths()
