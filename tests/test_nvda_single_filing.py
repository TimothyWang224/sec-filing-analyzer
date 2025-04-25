"""
Test script for processing a single NVDA filing from 2023.

This script processes a single NVDA filing to help debug the pipeline.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from sec_filing_analyzer.config import ETLConfig
from sec_filing_analyzer.data_retrieval import FilingProcessor, SECFilingsDownloader
from sec_filing_analyzer.data_retrieval.file_storage import FileStorage
from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
from sec_filing_analyzer.storage import GraphStore, LlamaIndexVectorStore

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set specific loggers to DEBUG level
logging.getLogger("sec_filing_analyzer").setLevel(logging.DEBUG)


def test_nvda_single_filing():
    """Test processing a single NVDA filing."""
    try:
        logger.info("Initializing ETL pipeline components")

        # Initialize storage components
        vector_store = LlamaIndexVectorStore()
        graph_store = GraphStore()

        # Initialize file storage
        file_storage = FileStorage(base_dir=ETLConfig().filings_dir)

        # Initialize SEC downloader
        sec_downloader = SECFilingsDownloader(file_storage=file_storage)

        # Initialize filing processor
        filing_processor = FilingProcessor(
            graph_store=graph_store, vector_store=vector_store, file_storage=file_storage
        )

        # Download a single NVDA filing
        logger.info("Downloading a single NVDA 10-K filing from 2023")

        # Use the SEC downloader directly
        from edgar import Company

        # Get NVDA company
        nvda = Company("NVDA")

        # Get 10-K filings from 2023
        filings = nvda.get_filings(
            form="10-K",
            date="2023-01-01:2023-12-31",  # Date range in the format expected by the library
        )

        if not filings:
            logger.error("No NVDA 10-K filings found for 2023")
            return

        # Get the first filing
        filing = filings[0]
        logger.info(f"Found filing: {filing.form} from {filing.filing_date}")

        # Download the filing
        filing_metadata = sec_downloader.download_filing(filing, "NVDA")

        if not filing_metadata:
            logger.error("Failed to download filing")
            return

        # Print the structure of the cached data
        logger.info(f"Cached data type: {type(filing_metadata)}")
        if isinstance(filing_metadata, dict):
            logger.info(f"Cached data keys: {list(filing_metadata.keys())}")
            for key, value in filing_metadata.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info(f"Cached data: {filing_metadata}")

        # Get the accession number from the filing object
        accession_number = filing.accession_number
        logger.info(f"Using accession number from filing object: {accession_number}")

        # Load the raw filing data
        filing_data = sec_downloader.get_filing(accession_number)

        if not filing_data:
            logger.error("Failed to load filing data")
            return

        logger.info(f"Successfully loaded filing data with {len(filing_data['content'])} characters")

        # Initialize pipeline
        pipeline = SECFilingETLPipeline(
            graph_store=graph_store,
            vector_store=vector_store,
            filing_processor=filing_processor,
            file_storage=file_storage,
            sec_downloader=sec_downloader,
        )

        # Process the filing
        logger.info(f"Processing filing: {accession_number}")

        # Create the expected filing data structure
        if isinstance(filing_metadata, dict) and "metadata" in filing_metadata:
            # Use the metadata from the cached data
            filing_data_for_processing = filing_metadata["metadata"]
        else:
            # Create metadata from the filing object
            filing_data_for_processing = {
                "accession_number": accession_number,
                "form": filing.form,
                "filing_date": filing.filing_date.isoformat(),
                "company": filing.company,
                "ticker": "NVDA",
                "cik": filing.cik,
                "has_html": True,
                "has_xml": False,
            }

        logger.info(f"Processing with filing data: {filing_data_for_processing}")
        pipeline.process_filing_data(filing_data_for_processing)

        logger.info("Filing processing completed successfully")

    except Exception as e:
        logger.error(f"Error processing filing: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    test_nvda_single_filing()
