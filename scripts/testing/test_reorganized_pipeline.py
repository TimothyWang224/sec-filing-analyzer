"""
Test script for the reorganized ETL pipeline.

This script demonstrates how to use the reorganized ETL pipeline with separate
semantic and quantitative data processing.
"""

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
from sec_filing_analyzer.pipeline.quantitative_pipeline import QuantitativeETLPipeline
from sec_filing_analyzer.pipeline.semantic_pipeline import SemanticETLPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def test_semantic_pipeline(ticker, filing_type, filing_date=None, accession_number=None):
    """
    Test the semantic pipeline.

    Args:
        ticker: Company ticker symbol
        filing_type: Type of filing (e.g., '10-K', '10-Q')
        filing_date: Date of filing (optional)
        accession_number: SEC accession number (optional)
    """
    try:
        # Create the semantic pipeline
        pipeline = SemanticETLPipeline()

        # Process a filing
        logger.info(f"Processing {filing_type} filing for {ticker} using semantic pipeline")
        result = pipeline.process_filing(
            ticker=ticker, filing_type=filing_type, filing_date=filing_date, accession_number=accession_number
        )

        # Print the result
        if "error" in result:
            logger.error(f"Error processing filing: {result['error']}")
        else:
            logger.info(f"Successfully processed filing: {result}")

        return result

    except Exception as e:
        logger.error(f"Error in test_semantic_pipeline: {e}")
        return {"error": str(e)}


def test_quantitative_pipeline(ticker, filing_type, filing_date=None, accession_number=None, db_path=None):
    """
    Test the quantitative pipeline.

    Args:
        ticker: Company ticker symbol
        filing_type: Type of filing (e.g., '10-K', '10-Q')
        filing_date: Date of filing (optional)
        accession_number: SEC accession number (optional)
        db_path: Path to the DuckDB database file
    """
    try:
        # Create the quantitative pipeline
        pipeline = QuantitativeETLPipeline(db_path=db_path)

        # Process a filing
        logger.info(f"Processing {filing_type} filing for {ticker} using quantitative pipeline")
        result = pipeline.process_filing(
            ticker=ticker, filing_type=filing_type, filing_date=filing_date, accession_number=accession_number
        )

        # Print the result
        if "error" in result:
            logger.error(f"Error processing filing: {result['error']}")
        else:
            logger.info(f"Successfully processed filing: {result}")

        return result

    except Exception as e:
        logger.error(f"Error in test_quantitative_pipeline: {e}")
        return {"error": str(e)}


def test_unified_pipeline(ticker, filing_type, filing_date=None, accession_number=None, db_path=None):
    """
    Test the unified ETL pipeline.

    Args:
        ticker: Company ticker symbol
        filing_type: Type of filing (e.g., '10-K', '10-Q')
        filing_date: Date of filing (optional)
        accession_number: SEC accession number (optional)
        db_path: Path to the DuckDB database file
    """
    try:
        # Create the unified pipeline
        pipeline = SECFilingETLPipeline(process_semantic=True, process_quantitative=True, db_path=db_path)

        # Process a filing
        logger.info(f"Processing {filing_type} filing for {ticker} using unified pipeline")
        result = pipeline.process_filing(
            ticker=ticker, filing_type=filing_type, filing_date=filing_date, accession_number=accession_number
        )

        # Print the result
        if "error" in result:
            logger.error(f"Error processing filing: {result['error']}")
        else:
            logger.info(f"Successfully processed filing: {result}")

        return result

    except Exception as e:
        logger.error(f"Error in test_unified_pipeline: {e}")
        return {"error": str(e)}


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test the reorganized ETL pipeline")
    parser.add_argument(
        "--mode",
        choices=["semantic", "quantitative", "unified"],
        default="unified",
        help="Test mode: semantic pipeline, quantitative pipeline, or unified pipeline",
    )
    parser.add_argument("--ticker", default="MSFT", help="Company ticker symbol")
    parser.add_argument("--filing-type", default="10-K", help="Filing type (e.g., 10-K, 10-Q)")
    parser.add_argument("--filing-date", help="Filing date (optional)")
    parser.add_argument("--accession", help="SEC accession number (optional)")
    parser.add_argument("--db-path", help="Path to the DuckDB database file")

    args = parser.parse_args()

    # Set default database path if not provided
    db_path = args.db_path or "data/financial_data.duckdb"

    # Ensure the database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Run the appropriate test based on the mode
    if args.mode == "semantic":
        test_semantic_pipeline(
            ticker=args.ticker,
            filing_type=args.filing_type,
            filing_date=args.filing_date,
            accession_number=args.accession,
        )

    elif args.mode == "quantitative":
        test_quantitative_pipeline(
            ticker=args.ticker,
            filing_type=args.filing_type,
            filing_date=args.filing_date,
            accession_number=args.accession,
            db_path=db_path,
        )

    elif args.mode == "unified":
        test_unified_pipeline(
            ticker=args.ticker,
            filing_type=args.filing_type,
            filing_date=args.filing_date,
            accession_number=args.accession,
            db_path=db_path,
        )


if __name__ == "__main__":
    # Set edgar identity from environment variables
    edgar_identity = os.getenv("EDGAR_IDENTITY")
    if edgar_identity:
        import edgar

        edgar.set_identity(edgar_identity)
        logger.info(f"Set edgar identity to: {edgar_identity}")
    else:
        logger.warning("EDGAR_IDENTITY environment variable not set. Set it in your .env file.")

    main()
