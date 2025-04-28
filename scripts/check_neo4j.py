"""
Script to check Neo4j database structure and content
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.sec_filing_analyzer.storage.graph_store import GraphStore

    def check_neo4j():
        """Check Neo4j database structure and content."""
        try:
            # Initialize GraphStore
            graph_store = GraphStore(use_neo4j=True)

            # Check if connected
            print("Connected to Neo4j:", graph_store._driver is not None)

            # Get node counts by label
            print("\nNode counts by label:")
            labels = ["Company", "Filing", "Section", "Fact", "Metric"]
            for label in labels:
                count = graph_store.run_query(
                    f"MATCH (n:{label}) RETURN count(n) as count"
                )
                if count:
                    print(f"{label}: {count[0]['count']}")
                else:
                    print(f"{label}: 0")

            # Get relationship counts
            print("\nRelationship counts:")
            relationships = ["FILED", "CONTAINS", "HAS_VALUE", "RELATED_TO"]
            for rel in relationships:
                count = graph_store.run_query(
                    f"MATCH ()-[r:{rel}]->() RETURN count(r) as count"
                )
                if count:
                    print(f"{rel}: {count[0]['count']}")
                else:
                    print(f"{rel}: 0")

            # Get sample companies
            print("\nSample companies:")
            companies = graph_store.run_query(
                "MATCH (c:Company) RETURN c.ticker as ticker, c.name as name LIMIT 5"
            )
            for company in companies:
                print(f"{company['ticker']} - {company['name']}")

            # Get sample filings
            print("\nSample filings:")
            filings = graph_store.run_query("""
                MATCH (c:Company)-[:FILED]->(f:Filing) 
                RETURN c.ticker as ticker, f.filing_type as type, f.filing_date as date, 
                       f.accession_number as accession_number, f.file_path as file_path
                LIMIT 5
            """)
            for filing in filings:
                print(
                    f"{filing['ticker']} - {filing['type']} - {filing['date']} - {filing.get('file_path', 'No path')}"
                )

            # Check if file paths are stored
            print("\nChecking for file paths in Neo4j:")
            file_paths = graph_store.run_query("""
                MATCH (n) 
                WHERE n.file_path IS NOT NULL 
                RETURN labels(n) as labels, n.file_path as file_path 
                LIMIT 10
            """)
            if file_paths:
                for path in file_paths:
                    print(f"{path['labels']}: {path['file_path']}")
            else:
                print("No file paths found in Neo4j")

        except Exception as e:
            print(f"Error connecting to Neo4j: {e}")

    if __name__ == "__main__":
        check_neo4j()

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the project root directory.")
