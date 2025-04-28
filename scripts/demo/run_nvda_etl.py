#!/usr/bin/env python
"""
Run ETL pipeline for NVIDIA Corporation (NVDA).

This script processes SEC filings for NVIDIA Corporation (NVDA) for the specified years.
It downloads the filings, extracts the data, and stores it in the database.

Usage:
    python scripts/demo/run_nvda_etl.py --ticker NVDA --years 2023 2024

    # Use synthetic data for testing
    TEST_MODE=True python scripts/demo/run_nvda_etl.py --ticker NVDA --years 2023 2024
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the actual ETL pipeline components
from sec_filing_analyzer.config import ETLConfig
from sec_filing_analyzer.data_retrieval import SECFilingsDownloader
from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
from sec_filing_analyzer.storage import GraphStore, LlamaIndexVectorStore


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process SEC filings for NVIDIA Corporation")
    parser.add_argument(
        "--ticker",
        type=str,
        default="NVDA",
        help="Company ticker symbol (default: NVDA)"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Years to process (e.g., 2023 2024)"
    )
    parser.add_argument(
        "--filing-types",
        nargs="+",
        default=["10-K", "10-Q"],
        help="Filing types to process (default: 10-K 10-Q)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/db_backup/financial_data.duckdb",
        help="Path to DuckDB database"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use synthetic data instead of downloading real filings"
    )

    return parser.parse_args()


def setup_synthetic_mode():
    """Set up the environment for synthetic data mode."""
    logger.info("Setting up synthetic data mode")

    # Create synthetic data directory if it doesn't exist
    os.makedirs("data/synthetic", exist_ok=True)

    # Create a minimal synthetic filing if it doesn't exist
    synthetic_file = Path("data/synthetic/nvda_stub.txt")
    if not synthetic_file.exists():
        logger.info("Creating synthetic filing data")
        with open(synthetic_file, "w") as f:
            f.write("""<DOCUMENT>
<TYPE>10-K
<SEQUENCE>1
<FILENAME>nvda-20240128x10k.htm
<DESCRIPTION>10-K
<TEXT>
<HTML>
<HEAD>
<TITLE>NVIDIA CORPORATION 10-K</TITLE>
</HEAD>
<BODY>

<H1>NVIDIA CORPORATION</H1>
<H2>FORM 10-K</H2>
<H3>Annual Report Pursuant to Section 13 or 15(d) of the Securities Exchange Act of 1934</H3>
<P>For the fiscal year ended January 28, 2024</P>

<H2>PART I</H2>

<H3>ITEM 1. BUSINESS</H3>

<P>NVIDIA Corporation is the inventor of the GPU, a powerful processor that has transformed computing in ways that were previously unimaginable. Our invention of the GPU in 1999 sparked the growth of the PC gaming market and redefined computer graphics. More recently, GPU deep learning has ignited modern AI, the next era of computing.</P>

<H3>ITEM 6. SELECTED FINANCIAL DATA</H3>

<TABLE>
<TR>
<TH>Revenue (in millions)</TH>
<TH>2024</TH>
<TH>2023</TH>
<TH>2022</TH>
</TR>
<TR>
<TD>Total Revenue</TD>
<TD>$26,974</TD>
<TD>$26,974</TD>
<TD>$16,675</TD>
</TR>
</TABLE>

<H3>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</H3>

<P>Revenue for fiscal year 2024 was $26.97 billion, up 126% from a year earlier.</P>

</BODY>
</HTML>
</TEXT>
</DOCUMENT>""")

    # For the demo, we'll use a simpler approach
    # Instead of patching the edgar module, we'll use environment variables
    # that the SECFilingsDownloader will check

    # Set environment variables for synthetic mode
    os.environ["SEC_USE_SYNTHETIC_DATA"] = "True"
    os.environ["SEC_SYNTHETIC_DATA_PATH"] = str(synthetic_file.absolute())

    logger.info("Synthetic data mode setup complete")


def main():
    """Run the ETL pipeline for NVIDIA Corporation."""
    # Load environment variables
    load_dotenv()

    # Parse command-line arguments
    args = parse_args()

    # Check if we should use synthetic data
    test_mode = args.test_mode or os.environ.get("TEST_MODE", "False").lower() in ("true", "1", "t")

    # Check for OPENAI_API_KEY (only needed for real processing)
    if not test_mode and not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        logger.info("Please set OPENAI_API_KEY in your .env file or environment")
        logger.info("Alternatively, use --test-mode to run with synthetic data")
        return 1

    # Set up synthetic mode if needed
    if test_mode:
        setup_synthetic_mode()
        # Import the patch for SECFilingsDownloader
        import scripts.demo.sec_downloader_patch

    logger.info(f"Starting ETL process for {args.ticker}")
    logger.info(f"Processing years: {', '.join(map(str, args.years))}")
    logger.info(f"Mode: {'TEST (synthetic data)' if test_mode else 'PRODUCTION (real data)'}")

    # Create date ranges for each year
    date_ranges = []
    for year in args.years:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        date_ranges.append((start_date, end_date))

    # Initialize SEC downloader
    downloader = SECFilingsDownloader()

    # Initialize a simplified ETL pipeline
    logger.info("Initializing ETL pipeline...")
    pipeline = SECFilingETLPipeline(
        sec_downloader=downloader,
        process_semantic=True,
        process_quantitative=True,
        db_path=args.db_path,
    )

    # Process filings for each date range
    results = []
    for start_date, end_date in date_ranges:
        logger.info(f"Processing filings from {start_date} to {end_date}")

        try:
            # Process company filings
            result = pipeline.process_company(
                ticker=args.ticker,
                filing_types=args.filing_types,
                start_date=start_date,
                end_date=end_date,
            )

            results.append(result)

            if result.get("status") == "success":
                logger.info(f"Successfully processed {args.ticker} filings from {start_date} to {end_date}")
            else:
                logger.warning(f"Partially processed {args.ticker} filings from {start_date} to {end_date}")
                logger.warning(f"Errors: {result.get('errors', [])}")

        except Exception as e:
            logger.error(f"Error processing {args.ticker} filings from {start_date} to {end_date}: {str(e)}")
            results.append({"status": "error", "error": str(e)})

    # Summarize results
    success_count = sum(1 for r in results if r.get("status") == "success")
    partial_count = sum(1 for r in results if r.get("status") == "partial")
    error_count = sum(1 for r in results if r.get("status") == "error")

    logger.info(f"ETL process completed for {args.ticker}")
    logger.info(f"Success: {success_count}, Partial: {partial_count}, Error: {error_count}")

    return 0


if __name__ == "__main__":
    exit(main())
