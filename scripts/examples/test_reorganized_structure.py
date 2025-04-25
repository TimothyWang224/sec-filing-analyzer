"""
Test script for the reorganized directory structure.

This script tests that the reorganized directory structure is working correctly
by importing and instantiating classes from the new locations.
"""

import logging
import os

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def test_semantic_imports():
    """Test importing classes from the semantic module."""
    try:
        # Import classes from the semantic module
        from sec_filing_analyzer.semantic.embeddings.embedding_generator import EmbeddingGenerator
        from sec_filing_analyzer.semantic.processing.chunking import DocumentChunker
        from sec_filing_analyzer.semantic.storage.vector_store import VectorStore

        # Instantiate classes
        chunker = DocumentChunker()
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore()

        logger.info("Successfully imported and instantiated classes from the semantic module")
        return True
    except Exception as e:
        logger.error(f"Error importing classes from the semantic module: {e}")
        return False


def test_quantitative_imports():
    """Test importing classes from the quantitative module."""
    try:
        # Import classes from the quantitative module
        from sec_filing_analyzer.quantitative.processing.edgar_xbrl_to_duckdb import EdgarXBRLToDuckDBExtractor
        from sec_filing_analyzer.quantitative.storage.optimized_duckdb_store import OptimizedDuckDBStore

        # Instantiate classes
        db_path = "data/test_financial_data.duckdb"
        duckdb_store = OptimizedDuckDBStore(db_path=db_path)
        xbrl_extractor = EdgarXBRLToDuckDBExtractor(db_path=db_path)

        logger.info("Successfully imported and instantiated classes from the quantitative module")
        return True
    except Exception as e:
        logger.error(f"Error importing classes from the quantitative module: {e}")
        return False


def test_pipeline_imports():
    """Test importing classes from the pipeline module."""
    try:
        # Import classes from the pipeline module
        from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
        from sec_filing_analyzer.pipeline.quantitative_pipeline import QuantitativeETLPipeline
        from sec_filing_analyzer.pipeline.semantic_pipeline import SemanticETLPipeline

        # Instantiate classes
        semantic_pipeline = SemanticETLPipeline()
        quantitative_pipeline = QuantitativeETLPipeline()
        etl_pipeline = SECFilingETLPipeline()

        logger.info("Successfully imported and instantiated classes from the pipeline module")
        return True
    except Exception as e:
        logger.error(f"Error importing classes from the pipeline module: {e}")
        return False


def main():
    """Main function to run the test script."""
    logger.info("Testing reorganized directory structure")

    # Test semantic imports
    semantic_result = test_semantic_imports()

    # Test quantitative imports
    quantitative_result = test_quantitative_imports()

    # Test pipeline imports
    pipeline_result = test_pipeline_imports()

    # Print summary
    logger.info("Test results:")
    logger.info(f"  Semantic imports: {'Success' if semantic_result else 'Failure'}")
    logger.info(f"  Quantitative imports: {'Success' if quantitative_result else 'Failure'}")
    logger.info(f"  Pipeline imports: {'Success' if pipeline_result else 'Failure'}")

    # Overall result
    if semantic_result and quantitative_result and pipeline_result:
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed")


if __name__ == "__main__":
    main()
