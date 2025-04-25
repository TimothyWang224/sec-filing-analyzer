"""
Test script for the optimized parallel processor.
"""

import argparse
import json
import logging
from datetime import datetime, timedelta

from src.sec_filing_analyzer.pipeline.optimized_parallel_processor import OptimizedParallelProcessor
from src.sec_filing_analyzer.utils.etl_logging import generate_run_id, setup_etl_logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set up ETL logging
setup_etl_logging()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test the optimized parallel processor")

    # Company selection
    parser.add_argument(
        "--tickers", type=str, nargs="+", default=["AAPL", "MSFT", "NVDA"], help="Ticker symbols to process"
    )

    # Filing types
    parser.add_argument("--filing-types", type=str, nargs="+", default=["8-K"], help="Filing types to process")

    # Date range
    parser.add_argument("--days", type=int, default=30, help="Number of days to look back for filings")
    parser.add_argument("--start-date", type=str, help="Start date for filings (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for filings (YYYY-MM-DD)")

    # Processing options
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of filings to process per company")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding generation")
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Initial rate limit in seconds")
    parser.add_argument(
        "--force-reprocess", action="store_true", help="Force reprocessing of already processed filings"
    )
    parser.add_argument(
        "--max-companies", type=int, default=2, help="Maximum number of companies to process in parallel"
    )

    # Processing mode
    parser.add_argument("--semantic-only", action="store_true", help="Only process semantic data")
    parser.add_argument("--quantitative-only", action="store_true", help="Only process quantitative data")

    # Output options
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    return parser.parse_args()


def main():
    """Run the test."""
    args = parse_args()

    # Generate a run ID
    run_id = generate_run_id()

    # Set up date range
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        today = datetime.now()
        start_date = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

    logger.info(f"Processing filings from {start_date} to {end_date}")

    # Determine processing modes
    process_semantic = not args.quantitative_only
    process_quantitative = not args.semantic_only

    if args.semantic_only and args.quantitative_only:
        logger.warning("Both --semantic-only and --quantitative-only specified, processing both")
        process_semantic = True
        process_quantitative = True

    # Initialize processor
    processor = OptimizedParallelProcessor(
        max_workers=args.max_workers,
        initial_rate_limit=args.rate_limit,
        batch_size=args.batch_size,
        process_semantic=process_semantic,
        process_quantitative=process_quantitative,
    )

    # Process companies
    results = processor.process_multiple_companies(
        tickers=args.tickers,
        filing_types=args.filing_types,
        start_date=start_date,
        end_date=end_date,
        limit_per_company=args.limit,
        force_reprocess=args.force_reprocess,
        max_companies_in_parallel=args.max_companies,
        run_id=run_id,
    )

    # Print summary
    logger.info("Processing complete!")
    logger.info(f"Overall stats: {json.dumps(results['stats'], indent=2)}")
    logger.info(f"Rate limiter stats: {json.dumps(results['rate_limiter_stats'], indent=2)}")
    logger.info(f"Run ID: {run_id} - Check the ETL logs for detailed information")

    # Save results if output file specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
