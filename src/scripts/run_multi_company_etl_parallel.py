"""
Script to run ETL process for SEC filings for multiple companies in parallel
"""

import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import sys
import time
import json
import concurrent.futures
import threading

# Import from the correct package path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  # Add root to path

from sec_filing_analyzer.pipeline.parallel_etl_pipeline import ParallelSECFilingETLPipeline
from sec_filing_analyzer.config import ETLConfig, StorageConfig, Neo4jConfig
from sec_filing_analyzer.storage.graph_store import GraphStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-local storage for pipeline instances
thread_local = threading.local()

def get_pipeline(use_neo4j=True, neo4j_config=None):
    """Get or create a thread-local pipeline instance."""
    if not hasattr(thread_local, "pipeline"):
        # Initialize graph store
        graph_store = None
        if use_neo4j:
            graph_store = GraphStore(
                use_neo4j=True,
                username=neo4j_config.get("username"),
                password=neo4j_config.get("password"),
                url=neo4j_config.get("url"),
                database=neo4j_config.get("database")
            )
        else:
            graph_store = GraphStore(use_neo4j=False)

        # Initialize pipeline
        thread_local.pipeline = ParallelSECFilingETLPipeline(graph_store=graph_store, max_workers=4)

    return thread_local.pipeline

def get_neo4j_config():
    """Get Neo4j configuration from environment variables."""
    return {
        "username": os.environ.get("NEO4J_USERNAME", "neo4j"),
        "password": os.environ.get("NEO4J_PASSWORD", "password"),
        "url": os.environ.get("NEO4J_URL", "bolt://localhost:7687"),
        "database": os.environ.get("NEO4J_DATABASE", "neo4j")
    }

def parse_args():
    neo4j_config = get_neo4j_config()
    parser = argparse.ArgumentParser(description='Process SEC filings for multiple companies in parallel')

    # Company tickers argument - either from file or direct list
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument('--tickers', nargs='+',
                       help='List of company ticker symbols (e.g., AAPL MSFT NVDA)')
    ticker_group.add_argument('--tickers-file', type=str,
                       help='Path to a JSON file containing a list of ticker symbols')

    # Date range arguments
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)', required=True)
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)', required=True)

    # Filing types argument
    parser.add_argument('--filing-types', nargs='+',
                       help='List of filing types to process (e.g., 10-K 10-Q)',
                       default=['10-K', '10-Q'])

    # Neo4j arguments
    parser.add_argument('--no-neo4j', action='store_true',
                       help='Use in-memory graph store instead of Neo4j')
    parser.add_argument('--neo4j-username', type=str,
                       default=neo4j_config["username"],
                       help='Neo4j username')
    parser.add_argument('--neo4j-password', type=str,
                       default=neo4j_config["password"],
                       help='Neo4j password')
    parser.add_argument('--neo4j-url', type=str,
                       default=neo4j_config["url"],
                       help='Neo4j URL')
    parser.add_argument('--neo4j-database', type=str,
                       default=neo4j_config["database"],
                       help='Neo4j database name')

    # Additional options
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry failed companies from previous runs')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum number of retries for failed companies')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of worker threads')
    parser.add_argument('--rate-limit', type=float, default=0.5,
                       help='Minimum time between API requests in seconds')

    return parser.parse_args()

def validate_dates(start_date, end_date):
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

def get_tickers_from_file(file_path):
    """Load ticker symbols from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'tickers' in data:
            return data['tickers']
        else:
            logger.error(f"Invalid format in {file_path}. Expected a list or a dict with 'tickers' key")
            return []
    except Exception as e:
        logger.error(f"Error loading tickers from {file_path}: {str(e)}")
        return []

def load_latest_progress():
    """Load the latest progress file."""
    try:
        progress_dir = Path('data/etl_progress')
        if not progress_dir.exists():
            return None

        # Find the latest progress file
        progress_files = list(progress_dir.glob('etl_progress_*.json'))
        if not progress_files:
            return None

        latest_file = max(progress_files, key=lambda x: x.stat().st_mtime)

        # Load progress
        with open(latest_file, 'r') as f:
            progress = json.load(f)

        return progress
    except Exception as e:
        logger.error(f"Error loading progress: {str(e)}")
        return None

def save_progress(completed: List[str], failed: List[str], no_filings: List[str], errors: dict):
    """Save progress to a file."""
    progress = {
        'timestamp': datetime.now().isoformat(),
        'completed': completed,
        'failed': failed,
        'no_filings': no_filings,
        'errors': errors
    }

    # Create directory if it doesn't exist
    os.makedirs('data/etl_progress', exist_ok=True)

    # Save progress to file
    filename = f"etl_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(f"data/etl_progress/{filename}", 'w') as f:
        json.dump(progress, f, indent=2)

    logger.info(f"Progress saved to data/etl_progress/{filename}")

def process_company(ticker, filing_types, start_date, end_date, args, rate_limiter):
    """Process a single company with rate limiting."""
    # Get thread-local pipeline
    neo4j_config = {
        "username": args.neo4j_username,
        "password": args.neo4j_password,
        "url": args.neo4j_url,
        "database": args.neo4j_database
    }
    pipeline = get_pipeline(not args.no_neo4j, neo4j_config)

    retries = 0
    result = {
        "ticker": ticker,
        "status": "failed",
        "filings_processed": 0,
        "error": None
    }

    while retries <= args.max_retries:
        try:
            logger.info(f"Processing company {ticker} (attempt {retries + 1}/{args.max_retries + 1})")

            # Apply rate limiting
            with rate_limiter:
                # Download filings
                downloaded_filings = pipeline.sec_downloader.download_company_filings(
                    ticker=ticker,
                    filing_types=filing_types,
                    start_date=start_date,
                    end_date=end_date
                )

            if not downloaded_filings:
                logger.warning(f"No filings found for {ticker} in the specified date range and filing types")
                result["status"] = "no_filings"
                return result

            logger.info(f"Found {len(downloaded_filings)} filings for {ticker}")

            # Process each filing
            processed_count = 0
            for filing_data in downloaded_filings:
                try:
                    pipeline.process_filing_data(filing_data)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing filing {filing_data['accession_number']}: {e}")

            logger.info(f"Successfully processed {processed_count} filings for {ticker}")
            result["status"] = "completed"
            result["filings_processed"] = processed_count
            return result

        except Exception as e:
            retries += 1
            error_msg = f"Error processing company {ticker}: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg

            if retries <= args.max_retries:
                logger.info(f"Retrying {ticker} in 5 seconds...")
                time.sleep(5)  # Wait before retrying

    logger.error(f"Failed to process company {ticker} after {args.max_retries + 1} attempts")
    return result

class RateLimiter:
    """Simple rate limiter to prevent API throttling."""
    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.lock = threading.Lock()

    def __enter__(self):
        with self.lock:
            # Calculate time to wait
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.rate_limit:
                time_to_wait = self.rate_limit - time_since_last
                time.sleep(time_to_wait)

            # Update last request time
            self.last_request_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def main():
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
    logger.info(f"Using {args.max_workers} worker threads")
    logger.info(f"Rate limit: {args.rate_limit} seconds between requests")

    # Track progress
    completed_tickers = []
    failed_tickers = []
    no_filings_tickers = []
    errors = {}

    # Load previous progress if retrying failed companies
    if args.retry_failed:
        progress = load_latest_progress()
        if progress:
            logger.info(f"Loaded previous progress with {len(progress['failed'])} failed companies")
            # Only process previously failed companies
            tickers = progress['failed']
            # Keep track of previously completed companies
            completed_tickers = progress['completed']
            # Keep track of companies with no filings
            if 'no_filings' in progress:
                no_filings_tickers = progress['no_filings']
            # Keep track of previous errors
            errors = progress['errors']

    # Create rate limiter
    rate_limiter = RateLimiter(args.rate_limit)

    # Process companies in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit tasks
        future_to_ticker = {
            executor.submit(
                process_company,
                ticker,
                args.filing_types,
                args.start_date,
                args.end_date,
                args,
                rate_limiter
            ): ticker for ticker in tickers
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result["status"] == "completed":
                    completed_tickers.append(ticker)
                    logger.info(f"Successfully processed {ticker} with {result['filings_processed']} filings")
                elif result["status"] == "no_filings":
                    no_filings_tickers.append(ticker)
                    logger.info(f"No filings found for {ticker}")
                else:
                    failed_tickers.append(ticker)
                    errors[ticker] = result["error"]
                    logger.error(f"Failed to process {ticker}: {result['error']}")
            except Exception as e:
                failed_tickers.append(ticker)
                errors[ticker] = str(e)
                logger.error(f"Exception processing {ticker}: {str(e)}")

    # Save progress
    save_progress(completed_tickers, failed_tickers, no_filings_tickers, errors)

    # Print summary
    logger.info("ETL process completed")
    logger.info(f"Successfully processed {len(completed_tickers)} companies")
    logger.info(f"No filings found for {len(no_filings_tickers)} companies")
    logger.info(f"Failed to process {len(failed_tickers)} companies")

    if no_filings_tickers:
        logger.info(f"Companies with no filings: {', '.join(no_filings_tickers)}")
        logger.info("Consider using a different date range or filing types for these companies")

    if failed_tickers:
        logger.info(f"Failed companies: {', '.join(failed_tickers)}")
        logger.info("To retry failed companies, run with --retry-failed flag")

if __name__ == "__main__":
    main()
