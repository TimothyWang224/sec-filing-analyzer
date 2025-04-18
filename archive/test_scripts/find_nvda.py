import os
import sys
import duckdb

def check_for_nvda(db_path):
    """Check if a DuckDB database contains NVDA data."""
    try:
        # Connect to the database in read-only mode
        conn = duckdb.connect(db_path, read_only=True)
        
        # Check if the companies table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [table[0] for table in tables]
        
        if 'companies' not in table_names:
            conn.close()
            return False
        
        # Check for NVDA ticker
        result = conn.execute("SELECT COUNT(*) FROM companies WHERE ticker = 'NVDA'").fetchone()
        has_nvda = result[0] > 0
        
        if has_nvda:
            print(f"\nFound NVDA in {db_path}")
            # Get filing count
            filings = conn.execute("""
                SELECT COUNT(*) 
                FROM filings f
                JOIN companies c ON f.company_id = c.company_id
                WHERE c.ticker = 'NVDA'
            """).fetchone()
            print(f"NVDA has {filings[0]} filings in this database")
        
        conn.close()
        return has_nvda
    except Exception as e:
        print(f"Error checking {db_path}: {e}")
        return False

def find_duckdb_files(start_dir='.'):
    """Find all .duckdb files in the directory tree."""
    duckdb_files = []
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith('.duckdb'):
                duckdb_files.append(os.path.join(root, file))
    return duckdb_files

def main():
    """Main function."""
    # Find all DuckDB files
    duckdb_files = find_duckdb_files()
    print(f"Found {len(duckdb_files)} DuckDB files")
    
    # Check each file for NVDA data
    nvda_files = []
    for db_path in duckdb_files:
        try:
            if check_for_nvda(db_path):
                nvda_files.append(db_path)
        except Exception as e:
            print(f"Error processing {db_path}: {e}")
    
    # Print results
    if nvda_files:
        print(f"\nFound NVDA data in {len(nvda_files)} files:")
        for file in nvda_files:
            print(f"  - {file}")
    else:
        print("\nNo NVDA data found in any DuckDB file")

if __name__ == "__main__":
    main()
