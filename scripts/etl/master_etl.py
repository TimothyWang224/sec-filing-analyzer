"""
Master ETL Script for SEC Filing Analysis

This script provides a centralized entry point for the complete ETL process:
1. Retrieving SEC filings using edgar
2. Chunking, embedding, and upserting to a vector store
3. Indexing the vectors in FAISS with optimized parameters
4. Creating a graph over the chunks and filings in Neo4j
5. Extracting XBRL facts and storing them in DuckDB

Usage:
    python scripts/etl/master_etl.py --tickers AAPL MSFT --start-date 2022-01-01 --end-date 2023-01-01
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sec_filing_analyzer.config import StorageConfig
from sec_filing_analyzer.pipeline.parallel_etl_pipeline import (
    ParallelSECFilingETLPipeline,
)
from sec_filing_analyzer.storage.graph_store import GraphStore
from sec_filing_analyzer.storage.optimized_vector_store import OptimizedVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_neo4j_config():
    """Get Neo4j configuration from environment variables."""
    return {
        "url": os.getenv("NEO4J_URL", "bolt://localhost:7687"),
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "password"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }


def validate_dates(start_date: str, end_date: str) -> bool:
    """Validate date format and range."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        if start > end:
            logger.error("Start date must be before end date")
            return False
        return True
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD")
        return False


def get_tickers_from_file(file_path: str) -> List[str]:
    """Load ticker symbols from a JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "tickers" in data:
            return data["tickers"]
        else:
            logger.error(f"Invalid format in tickers file: {file_path}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading tickers file: {str(e)}")
        sys.exit(1)


def save_progress(
    completed: List[str], failed: List[str], no_filings: List[str], errors: dict
):
    """Save progress to a file."""
    progress = {
        "timestamp": datetime.now().isoformat(),
        "completed": completed,
        "failed": failed,
        "no_filings": no_filings,
        "errors": errors,
    }

    # Create directory if it doesn't exist
    os.makedirs("data/etl_progress", exist_ok=True)

    # Save progress to file
    filename = f"etl_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(f"data/etl_progress/{filename}", "w") as f:
        json.dump(progress, f, indent=2)

    logger.info(f"Progress saved to data/etl_progress/{filename}")


def load_latest_progress():
    """Load the latest progress file."""
    try:
        progress_dir = Path("data/etl_progress")
        if not progress_dir.exists():
            return None

        progress_files = list(progress_dir.glob("etl_progress_*.json"))
        if not progress_files:
            return None

        # Sort by modification time (newest first)
        latest_file = max(progress_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading progress file: {str(e)}")
        return None


def save_config(config: Dict[str, Any], config_path: Optional[str] = None):
    """Save configuration to a file."""
    if config_path is None:
        config_path = "data/config/etl_config.json"

    # Create directory if it doesn't exist
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration saved to {config_path}")


def load_config(config_path: Optional[str] = None):
    """Load configuration from a file."""
    if config_path is None:
        config_path = "data/config/etl_config.json"

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading configuration file: {str(e)}")
        return None


def parse_args():
    """Parse command line arguments."""
    neo4j_config = get_neo4j_config()
    parser = argparse.ArgumentParser(
        description="Master ETL Script for SEC Filing Analysis"
    )

    # Input group - Company tickers
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        "--tickers",
        nargs="+",
        help="List of company ticker symbols (e.g., AAPL MSFT NVDA)",
    )
    ticker_group.add_argument(
        "--tickers-file",
        type=str,
        help="Path to a JSON file containing a list of ticker symbols",
    )

    # Date range arguments
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)", required=True)
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)", required=True)

    # Filing types argument
    parser.add_argument(
        "--filing-types",
        nargs="+",
        help="List of filing types to process (e.g., 10-K 10-Q)",
        default=["10-K", "10-Q"],
    )

    # Configuration file
    parser.add_argument(
        "--config-file", type=str, help="Path to a JSON configuration file"
    )
    parser.add_argument(
        "--save-config", action="store_true", help="Save the configuration to a file"
    )
    parser.add_argument(
        "--config-output", type=str, help="Path to save the configuration file"
    )

    # Neo4j configuration arguments
    parser.add_argument(
        "--no-neo4j",
        action="store_true",
        help="Disable Neo4j and use in-memory graph store instead",
    )
    parser.add_argument(
        "--neo4j-url", help="Neo4j server URL", default=neo4j_config["url"]
    )
    parser.add_argument(
        "--neo4j-username", help="Neo4j username", default=neo4j_config["username"]
    )
    parser.add_argument(
        "--neo4j-password", help="Neo4j password", default=neo4j_config["password"]
    )
    parser.add_argument(
        "--neo4j-database", help="Neo4j database name", default=neo4j_config["database"]
    )

    # DuckDB configuration
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/financial_data.duckdb",
        help="Path to the DuckDB database file",
    )

    # Pipeline configuration
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic processing (chunking, embedding, etc.)",
    )
    parser.add_argument(
        "--no-quantitative",
        action="store_true",
        help="Disable quantitative processing (XBRL extraction, etc.)",
    )

    # Parallel processing options
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker threads for parallel processing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.1,
        help="Minimum time between API requests in seconds",
    )

    # FAISS indexing options
    parser.add_argument(
        "--index-type",
        type=str,
        default="hnsw",
        choices=["flat", "ivf", "hnsw", "ivfpq"],
        help="FAISS index type to use",
    )

    # IVF parameters
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=None,
        help="IVF: Number of clusters (default: sqrt(num_vectors))",
    )
    parser.add_argument(
        "--ivf-nprobe",
        type=int,
        default=None,
        help="IVF: Number of clusters to visit during search",
    )

    # HNSW parameters
    parser.add_argument(
        "--hnsw-m", type=int, default=32, help="HNSW: Number of connections per element"
    )
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=400,
        help="HNSW: Size of dynamic list during construction",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=200,
        help="HNSW: Size of dynamic list during search",
    )

    # GPU acceleration
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration for FAISS if available",
    )

    # Additional options
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed companies from previous runs",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed companies",
    )
    parser.add_argument(
        "--delay-between-companies",
        type=int,
        default=1,
        help="Delay in seconds between processing companies to avoid rate limiting",
    )
    parser.add_argument(
        "--force-rebuild-index",
        action="store_true",
        help="Force rebuild of FAISS index even if it exists",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download of filings even if they exist in cache",
    )

    args = parser.parse_args()

    # Load configuration file if specified
    if args.config_file:
        config = load_config(args.config_file)
        if config:
            # Update args with values from config file (only if not explicitly set on command line)
            for key, value in config.items():
                if key in vars(args) and vars(args)[key] is None:
                    setattr(args, key, value)

    return args


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()

    # Validate dates
    if not validate_dates(args.start_date, args.end_date):
        sys.exit(1)

    # Get ticker symbols
    tickers = []
    if args.tickers:
        tickers = args.tickers
    elif args.tickers_file:
        tickers = get_tickers_from_file(args.tickers_file)

    logger.info(f"Processing {len(tickers)} companies: {', '.join(tickers)}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Filing types: {', '.join(args.filing_types)}")

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

    # Initialize vector store with optimized FAISS parameters
    vector_store = OptimizedVectorStore(
        store_path=StorageConfig().vector_store_path,
        index_type=args.index_type,
        use_gpu=args.use_gpu,
        nlist=args.ivf_nlist,
        nprobe=args.ivf_nprobe,
        m=args.hnsw_m,
        ef_construction=args.hnsw_ef_construction,
        ef_search=args.hnsw_ef_search,
    )

    # Initialize pipeline with parallel processing options
    pipeline = ParallelSECFilingETLPipeline(
        graph_store=graph_store,
        vector_store=vector_store,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        rate_limit=args.rate_limit,
        use_optimized_vector_store=True,
        process_semantic=not args.no_semantic,
        process_quantitative=not args.no_quantitative,
        db_path=args.db_path,
    )

    logger.info(f"Parallel processing: {not args.no_parallel}")
    if not args.no_parallel:
        logger.info(f"Using {args.max_workers} worker threads")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Rate limit: {args.rate_limit} seconds")

    # Log FAISS configuration
    logger.info(f"FAISS index type: {args.index_type}")
    if args.index_type == "ivf":
        nlist = args.ivf_nlist or "auto (sqrt of vectors)"
        nprobe = args.ivf_nprobe or "auto (10% of nlist)"
        logger.info(f"IVF parameters: nlist={nlist}, nprobe={nprobe}")
    elif args.index_type == "hnsw":
        logger.info(
            f"HNSW parameters: m={args.hnsw_m}, ef_construction={args.hnsw_ef_construction}, ef_search={args.hnsw_ef_search}"
        )
    elif args.index_type == "ivfpq":
        nlist = args.ivf_nlist or "auto (sqrt of vectors)"
        nprobe = args.ivf_nprobe or "auto (10% of nlist)"
        logger.info(f"IVFPQ parameters: nlist={nlist}, nprobe={nprobe}")

    # Save configuration if requested
    if args.save_config:
        config = {
            "vector_store": {
                "type": "optimized",
                "index_type": args.index_type,
                "use_gpu": args.use_gpu,
                "ivf_nlist": args.ivf_nlist,
                "ivf_nprobe": args.ivf_nprobe,
                "hnsw_m": args.hnsw_m,
                "hnsw_ef_construction": args.hnsw_ef_construction,
                "hnsw_ef_search": args.hnsw_ef_search,
                "path": str(StorageConfig().vector_store_path),
            },
            "graph_store": {
                "type": "neo4j" if not args.no_neo4j else "in-memory",
                "url": args.neo4j_url,
                "username": args.neo4j_username,
                "password": args.neo4j_password,
                "database": args.neo4j_database,
            },
            "duckdb": {"path": args.db_path},
            "etl_pipeline": {
                "process_semantic": not args.no_semantic,
                "process_quantitative": not args.no_quantitative,
                "use_parallel": not args.no_parallel,
                "max_workers": args.max_workers,
                "batch_size": args.batch_size,
                "rate_limit": args.rate_limit,
                "max_retries": args.max_retries,
                "delay_between_companies": args.delay_between_companies,
            },
        }
        save_config(config, args.config_output)

    # Track progress
    completed_tickers = []
    failed_tickers = []
    no_filings_tickers = []
    errors = {}

    # Load previous progress if retrying failed companies
    if args.retry_failed:
        progress = load_latest_progress()
        if progress:
            logger.info(
                f"Loaded previous progress with {len(progress['failed'])} failed companies"
            )
            # Only process previously failed companies
            tickers = progress["failed"]
            # Keep track of previously completed companies
            completed_tickers = progress["completed"]
            # Keep track of companies with no filings
            if "no_filings" in progress:
                no_filings_tickers = progress["no_filings"]
            # Keep track of previous errors
            errors = progress["errors"]

    # Process each company
    for ticker in tickers:
        retries = 0
        success = False

        while retries <= args.max_retries and not success:
            try:
                logger.info(
                    f"Processing company {ticker} (attempt {retries + 1}/{args.max_retries + 1})"
                )

                # Process company using the parallel pipeline
                result = pipeline.process_company(
                    ticker=ticker,
                    filing_types=args.filing_types,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    force_rebuild_index=args.force_rebuild_index,
                    force_download=args.force_download,
                )

                # Check result status
                if result["status"] == "no_filings":
                    logger.warning(
                        f"No filings found for {ticker} in the specified date range and filing types"
                    )
                    no_filings_tickers.append(ticker)
                    success = (
                        True  # Mark as success since there's no error, just no filings
                    )
                elif result["status"] == "completed":
                    logger.info(
                        f"Successfully processed {result['filings_processed']} filings for {ticker}"
                    )
                    completed_tickers.append(ticker)
                    success = True
                else:
                    # Failed processing
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Failed to process {ticker}: {error_msg}")
                    errors[ticker] = error_msg
                    retries += 1

            except Exception as e:
                retries += 1
                error_msg = f"Error processing company {ticker}: {str(e)}"
                logger.error(error_msg)
                errors[ticker] = error_msg

                if retries <= args.max_retries:
                    logger.info(f"Retrying {ticker} in 5 seconds...")
                    time.sleep(5)  # Wait before retrying

        if not success:
            logger.error(
                f"Failed to process company {ticker} after {args.max_retries + 1} attempts"
            )
            failed_tickers.append(ticker)

        # Add delay between companies to avoid rate limiting
        if args.delay_between_companies > 0 and ticker != tickers[-1]:
            logger.info(
                f"Waiting {args.delay_between_companies} seconds before processing next company..."
            )
            time.sleep(args.delay_between_companies)

    # Save progress
    save_progress(completed_tickers, failed_tickers, no_filings_tickers, errors)

    # Print summary
    logger.info("ETL process completed")
    logger.info(f"Successfully processed {len(completed_tickers)} companies")
    logger.info(f"No filings found for {len(no_filings_tickers)} companies")
    logger.info(f"Failed to process {len(failed_tickers)} companies")

    if no_filings_tickers:
        logger.info(f"Companies with no filings: {', '.join(no_filings_tickers)}")
        logger.info(
            "Consider using a different date range or filing types for these companies"
        )

    if failed_tickers:
        logger.info(f"Failed companies: {', '.join(failed_tickers)}")
        logger.info("To retry failed companies, run with --retry-failed flag")


if __name__ == "__main__":
    main()
