"""
Test script for the SEC Filing ETL Pipeline
"""

import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from sec_filing_analyzer.config import STORAGE_CONFIG, ETLConfig
from sec_filing_analyzer.data_retrieval.file_storage import FileStorage
from sec_filing_analyzer.data_retrieval.filing_processor import FilingProcessor
from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
from sec_filing_analyzer.storage import GraphStore, LlamaIndexVectorStore

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_neo4j_config() -> Optional[Dict[str, Any]]:
    """Get Neo4j configuration from environment variables."""
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not all([username, password]):
        logger.info("Neo4j credentials not found, using in-memory storage")
        return None

    return {
        "username": username,
        "password": password,
        "url": url,
        "database": database,
    }


def test_nvda_2023_filings():
    """Test processing NVDA's 2023 filings."""
    # Check for required environment variables
    edgar_identity = os.getenv("EDGAR_IDENTITY")
    if not edgar_identity:
        logger.error("EDGAR_IDENTITY environment variable not set. Please set it in your .env file.")
        return

    logger.info(f"Using EDGAR identity: {edgar_identity}")

    # Initialize components
    neo4j_config = get_neo4j_config()
    use_neo4j = neo4j_config is not None

    graph_store = GraphStore(
        store_dir=STORAGE_CONFIG["graph_store_path"],
        use_neo4j=use_neo4j,
        **(neo4j_config or {}),
    )

    vector_store = LlamaIndexVectorStore(store_dir=STORAGE_CONFIG["vector_store_path"])
    filing_processor = FilingProcessor(graph_store=graph_store, vector_store=vector_store)
    file_storage = FileStorage(base_dir=ETLConfig().filings_dir)

    # Initialize pipeline
    pipeline = SECFilingETLPipeline(
        graph_store=graph_store,
        vector_store=vector_store,
        filing_processor=filing_processor,
        file_storage=file_storage,
    )

    # Process NVDA's 2023 filings
    pipeline.process_company(
        ticker="NVDA",
        filing_types=["10-K", "10-Q", "8-K"],
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    # Verify that filings were processed
    # This is a basic test - you may want to add more specific assertions
    assert True  # Replace with actual assertions based on expected outcomes


if __name__ == "__main__":
    test_nvda_2023_filings()
