"""
Script to update the ETL pipeline to use the optimized vector store.

This script updates the ETL pipeline configuration to use the optimized vector store
and ensures that all necessary components are properly initialized.
"""

import json
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sec_filing_analyzer.config import StorageConfig
from sec_filing_analyzer.pipeline.parallel_etl_pipeline import (
    ParallelSECFilingETLPipeline,
)
from sec_filing_analyzer.search import CoordinatedSearch
from sec_filing_analyzer.storage import GraphStore, OptimizedVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def update_etl_pipeline():
    """Update the ETL pipeline to use the optimized vector store."""
    # Initialize the optimized vector store
    vector_store = OptimizedVectorStore(store_path=StorageConfig().vector_store_path)

    # Initialize the graph store
    graph_store = GraphStore(use_neo4j=True)

    # Initialize the ETL pipeline with the optimized vector store
    pipeline = ParallelSECFilingETLPipeline(
        graph_store=graph_store,
        vector_store=vector_store,
        max_workers=4,
        batch_size=50,  # Smaller batch size for better error handling
        rate_limit=0.2,  # Slightly higher rate limit to avoid API throttling
        use_optimized_vector_store=True,
    )

    # Initialize the coordinated search
    search = CoordinatedSearch(vector_store=vector_store, graph_store=graph_store)

    # Get vector store statistics
    stats = vector_store.get_stats()
    logger.info(f"Vector store statistics: {stats}")

    # Get available companies
    companies = []
    if graph_store.use_neo4j:
        try:
            with graph_store.driver.session(database=graph_store.database) as session:
                query = """
                MATCH (c:Company)
                RETURN c.ticker as ticker
                """
                result = session.run(query)
                companies = [record["ticker"] for record in result]
        except Exception as e:
            logger.error(f"Error getting companies from Neo4j: {e}")

    logger.info(f"Available companies: {companies}")

    # Save configuration
    config = {
        "vector_store": {
            "type": "optimized",
            "path": str(StorageConfig().vector_store_path),
            "stats": stats,
        },
        "graph_store": {
            "type": "neo4j" if graph_store.use_neo4j else "in-memory",
            "companies": companies,
        },
        "etl_pipeline": {"max_workers": 4, "batch_size": 50, "rate_limit": 0.2},
    }

    config_path = Path("data/config/etl_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"ETL pipeline configuration saved to {config_path}")
    logger.info("ETL pipeline updated to use the optimized vector store")


def main():
    """Main function to update the ETL pipeline."""
    logger.info("Updating ETL pipeline to use the optimized vector store")
    update_etl_pipeline()


if __name__ == "__main__":
    main()
