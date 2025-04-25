"""
Check the current state of the database and file system.
"""

import os
from pathlib import Path

import duckdb


def check_database():
    """Check the current state of the database."""
    try:
        # Connect to the database
        conn = duckdb.connect("data/db_backup/improved_financial_data.duckdb", read_only=True)

        # Check companies table
        print("Companies in DuckDB:")
        companies = conn.execute("SELECT * FROM companies").fetchdf()
        print(companies)

        # Check filings table
        print("\nFilings in DuckDB:")
        filings_count = conn.execute("SELECT COUNT(*) FROM filings_new").fetchone()[0]
        print(f"Total filings: {filings_count}")

        # Check company_id in filings
        company_ids = conn.execute("SELECT DISTINCT company_id FROM filings_new").fetchdf()
        print("Distinct company_ids in filings:")
        print(company_ids)

        # Get company names for each company_id
        print("\nCompany names for each company_id in filings:")
        for company_id in company_ids["company_id"]:
            company_name = conn.execute("SELECT name FROM companies WHERE company_id = ?", [company_id]).fetchone()
            if company_name:
                print(f"Company ID {company_id}: {company_name[0]}")
            else:
                print(f"Company ID {company_id}: Not found in companies table")

        # Close the connection
        conn.close()
    except Exception as e:
        print(f"Error checking database: {e}")


def check_embeddings():
    """Check the embeddings saved on disk."""
    try:
        # Check embeddings directory
        embeddings_dir = Path("data/embeddings")
        if not embeddings_dir.exists():
            print("\nEmbeddings directory not found")
            return

        print("\nEmbeddings on disk:")
        companies_with_embeddings = []

        # List all directories in the embeddings directory
        for item in embeddings_dir.iterdir():
            if item.is_dir():
                companies_with_embeddings.append(item.name)
                # Count number of embedding files
                embedding_files = list(item.glob("**/*.npy"))
                print(f"{item.name}: {len(embedding_files)} embedding files")

        if not companies_with_embeddings:
            print("No company embeddings found")
    except Exception as e:
        print(f"Error checking embeddings: {e}")


def check_raw_filings():
    """Check the raw filings saved on disk."""
    try:
        # Check raw filings directory
        raw_dir = Path("data/filings/raw")
        if not raw_dir.exists():
            print("\nRaw filings directory not found")
            return

        print("\nRaw filings on disk:")
        companies_with_filings = []

        # List all directories in the raw directory
        for item in raw_dir.iterdir():
            if item.is_dir():
                companies_with_filings.append(item.name)
                # Count number of filing files
                filing_files = list(item.glob("**/*.*"))
                print(f"{item.name}: {len(filing_files)} filing files")

        if not companies_with_filings:
            print("No company filings found")
    except Exception as e:
        print(f"Error checking raw filings: {e}")


if __name__ == "__main__":
    check_database()
    check_embeddings()
    check_raw_filings()
