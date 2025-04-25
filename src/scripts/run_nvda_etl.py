"""
Script to run ETL process for SEC filings
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sec_filing_analyzer.config import ETLConfig, Neo4jConfig, StorageConfig
from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
from sec_filing_analyzer.storage.graph_store import GraphStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_neo4j_config():
    """Get Neo4j configuration from environment variables or defaults."""
    config = Neo4jConfig()
    return {
        "url": os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI") or config.url,
        "username": os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or config.username,
        "password": os.getenv("NEO4J_PASSWORD") or config.password,
        "database": os.getenv("NEO4J_DATABASE") or config.database,
    }


def parse_args():
    neo4j_config = get_neo4j_config()
    parser = argparse.ArgumentParser(description="Process SEC filings for a company")
    parser.add_argument("ticker", help="Company ticker symbol (e.g., NVDA)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)", required=True)
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)", required=True)
    parser.add_argument(
        "--filing-types", nargs="+", help="List of filing types to process (e.g., 10-K 10-Q)", default=["10-K", "10-Q"]
    )
    parser.add_argument("--no-neo4j", action="store_true", help="Disable Neo4j and use in-memory graph store instead")
    parser.add_argument("--neo4j-url", help="Neo4j server URL", default=neo4j_config["url"])
    parser.add_argument("--neo4j-username", help="Neo4j username", default=neo4j_config["username"])
    parser.add_argument("--neo4j-password", help="Neo4j password", default=neo4j_config["password"])
    parser.add_argument("--neo4j-database", help="Neo4j database name", default=neo4j_config["database"])

    # Parallel processing options
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Maximum number of worker threads for parallel processing"
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding generation")
    parser.add_argument("--rate-limit", type=float, default=0.1, help="Minimum time between API requests in seconds")
    return parser.parse_args()


def validate_dates(start_date: str, end_date: str) -> None:
    """Validate date formats and ranges."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if end < start:
            raise ValueError("End date must be after start date")

        if end > datetime.now():
            raise ValueError("End date cannot be in the future")

    except ValueError as e:
        if "time data" in str(e):
            raise ValueError("Dates must be in YYYY-MM-DD format")
        raise


def main():
    # Parse command line arguments
    args = parse_args()

    # Validate dates
    validate_dates(args.start_date, args.end_date)

    # Initialize graph store (Neo4j by default, in-memory if --no-neo4j is specified)
    graph_store = None
    if args.no_neo4j:
        logger.info("Using in-memory graph store...")
        graph_store = GraphStore(use_neo4j=False)
    else:
        logger.info("Initializing Neo4j graph store...")
        graph_store = GraphStore(
            use_neo4j=True,
            username=args.neo4j_username,
            password=args.neo4j_password,
            url=args.neo4j_url,
            database=args.neo4j_database,
        )

    # Initialize pipeline with parallel processing options
    pipeline = SECFilingETLPipeline(
        graph_store=graph_store,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        rate_limit=args.rate_limit,
        use_parallel=not args.no_parallel,
    )

    logger.info(f"Parallel processing: {not args.no_parallel}")
    if not args.no_parallel:
        logger.info(f"Using {args.max_workers} worker threads")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Rate limit: {args.rate_limit} seconds")

    try:
        logger.info(f"Starting ETL process for {args.ticker}")
        logger.info(f"Date range: {args.start_date} to {args.end_date}")
        logger.info(f"Filing types: {', '.join(args.filing_types)}")
        logger.info(f"Graph store: {'In-memory' if args.no_neo4j else 'Neo4j'}")

        # Process company filings
        result = pipeline.process_company(
            ticker=args.ticker, filing_types=args.filing_types, start_date=args.start_date, end_date=args.end_date
        )

        # Check result status
        if result["status"] == "no_filings":
            logger.warning(f"No filings found for {args.ticker} in the specified date range and filing types")
        elif result["status"] == "completed":
            logger.info(f"Successfully processed {result['filings_processed']} filings for {args.ticker}")
        else:
            # Failed processing
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Failed to process {args.ticker}: {error_msg}")
            raise Exception(f"Failed to process {args.ticker}: {error_msg}")

        logger.info("ETL process completed successfully")

    except Exception as e:
        logger.error(f"Error running ETL process: {str(e)}")
        raise


if __name__ == "__main__":
    main()
