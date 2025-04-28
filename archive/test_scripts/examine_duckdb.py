import json

import duckdb


def examine_duckdb(db_path):
    """Examine the DuckDB database and return its structure and contents."""
    results = {}

    try:
        # Connect to the database in read-only mode
        con = duckdb.connect(db_path, read_only=True)

        # Get list of tables
        tables = con.execute("SHOW TABLES").fetchall()
        results["tables"] = [table[0] for table in tables]

        # For each table, get schema and sample data
        for table in results["tables"]:
            # Get schema
            schema = con.execute(f"DESCRIBE {table}").fetchall()
            results[f"{table}_schema"] = [
                {"column": col[0], "type": col[1]} for col in schema
            ]

            # Get row count
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            results[f"{table}_count"] = count

            # Get sample data (up to 10 rows)
            if count > 0:
                sample = con.execute(f"SELECT * FROM {table} LIMIT 10").fetchall()
                # Convert to list of dicts for better readability
                columns = [col[0] for col in schema]
                sample_dicts = []
                for row in sample:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        # Handle non-serializable types
                        try:
                            json.dumps(row[i])
                            row_dict[col] = row[i]
                        except (TypeError, OverflowError):
                            row_dict[col] = str(row[i])
                    sample_dicts.append(row_dict)
                results[f"{table}_sample"] = sample_dicts

        # Close the connection
        con.close()

    except Exception as e:
        results["error"] = str(e)

    return results


if __name__ == "__main__":
    db_path = "data/financial_data.duckdb"
    results = examine_duckdb(db_path)

    # Save results to file
    output_path = "duckdb_examination.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
