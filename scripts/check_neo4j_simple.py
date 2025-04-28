"""
Simple script to check Neo4j database
"""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    # Try to import Neo4j driver directly
    from neo4j import GraphDatabase

    def check_neo4j():
        """Check Neo4j database using direct connection."""
        try:
            # Default Neo4j connection parameters
            uri = "bolt://localhost:7687"
            username = "neo4j"
            password = "password"  # Default password, might need to be changed

            # Try to connect
            driver = GraphDatabase.driver(uri, auth=(username, password))
            logger.info(f"Connected to Neo4j at {uri}")

            # Check if connection works
            with driver.session() as session:
                # Get node counts
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]
                logger.info(f"Total nodes in database: {count}")

                # Get node counts by label
                logger.info("Node counts by label:")
                labels = ["Company", "Filing", "Section", "Chunk", "Topic"]
                for label in labels:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    label_count = result.single()["count"]
                    logger.info(f"  {label}: {label_count}")

                # Get sample companies
                logger.info("Sample companies:")
                result = session.run(
                    "MATCH (c:Company) RETURN c.ticker as ticker, c.name as name LIMIT 5"
                )
                for record in result:
                    logger.info(
                        f"  {record['ticker']} - {record.get('name', 'No name')}"
                    )

                # Get sample filings
                logger.info("Sample filings:")
                result = session.run("""
                    MATCH (c:Company)-[:FILED]->(f:Filing) 
                    RETURN c.ticker as ticker, f.filing_type as type, f.filing_date as date, 
                           f.accession_number as accession_number
                    LIMIT 5
                """)
                for record in result:
                    logger.info(
                        f"  {record['ticker']} - {record['type']} - {record.get('date', 'No date')}"
                    )

                # Check for file paths
                logger.info("Checking for file paths:")
                result = session.run("""
                    MATCH (n) 
                    WHERE EXISTS(n.file_path) 
                    RETURN labels(n)[0] as label, n.file_path as path 
                    LIMIT 10
                """)
                paths_found = False
                for record in result:
                    paths_found = True
                    logger.info(f"  {record['label']}: {record['path']}")

                if not paths_found:
                    logger.info("  No file paths found in Neo4j")

            driver.close()

        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")

    if __name__ == "__main__":
        check_neo4j()

except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure neo4j package is installed.")
