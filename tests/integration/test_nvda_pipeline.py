"""
Test script for running the data processing pipeline for NVDA filings from 2023.

This script processes NVDA's 2023 filings using the SEC Filing ETL Pipeline.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from sec_filing_analyzer.config import ETLConfig, StorageConfig
from sec_filing_analyzer.data_retrieval import FilingProcessor
from sec_filing_analyzer.data_retrieval.file_storage import FileStorage
from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
from sec_filing_analyzer.storage import GraphStore, LlamaIndexVectorStore

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set specific loggers to DEBUG level
logging.getLogger("sec_filing_analyzer").setLevel(logging.DEBUG)


def test_nvda_2023_filings():
    """Test processing NVDA's 2023 filings."""
    try:
        logger.info("Initializing ETL pipeline components")

        # Initialize storage components
        vector_store = LlamaIndexVectorStore()
        graph_store = GraphStore()

        # Initialize file storage
        file_storage = FileStorage(base_dir=ETLConfig().filings_dir)

        # Initialize filing processor
        filing_processor = FilingProcessor(
            graph_store=graph_store, vector_store=vector_store, file_storage=file_storage
        )

        # Initialize pipeline
        pipeline = SECFilingETLPipeline(
            graph_store=graph_store,
            vector_store=vector_store,
            filing_processor=filing_processor,
            file_storage=file_storage,
        )

        logger.info("Starting ETL process for NVDA")
        logger.info("Date range: 2023-01-01 to 2023-12-31")
        logger.info("Filing types: 10-K, 10-Q, 8-K")

        # Process NVDA's 2023 filings
        pipeline.process_company(
            ticker="NVDA", filing_types=["10-K", "10-Q", "8-K"], start_date="2023-01-01", end_date="2023-12-31"
        )

        logger.info("ETL process completed successfully")

    except Exception as e:
        logger.error(f"Error running ETL process: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    test_nvda_2023_filings()
