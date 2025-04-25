"""
Script to analyze token usage and API limits.

This script analyzes the token usage of the embedding generation process and compares it
to the OpenAI API limits to help optimize batch sizes and rate limits.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sec_filing_analyzer.config import ETLConfig
from sec_filing_analyzer.embeddings.parallel_embeddings import ParallelEmbeddingGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# OpenAI API limits for text-embedding-3-small
OPENAI_LIMITS = {
    "tokens_per_minute": 1_000_000,  # 1M TPM
    "requests_per_minute": 3_000,  # 3K RPM
    "tokens_per_day": 3_000_000,  # 3M TPD
}


def estimate_tokens_in_filing(filing_path: Path) -> int:
    """Estimate the number of tokens in a filing.

    Args:
        filing_path: Path to the filing JSON file

    Returns:
        Estimated token count
    """
    try:
        with open(filing_path, "r") as f:
            filing_data = json.load(f)

        if "processed_data" not in filing_data:
            return 0

        processed_data = filing_data["processed_data"]

        # Check if we have chunk texts
        if "chunk_texts" not in processed_data:
            return 0

        chunk_texts = processed_data["chunk_texts"]

        # Estimate tokens (1 token ≈ 4 characters for English text)
        char_count = sum(len(text) for text in chunk_texts if text)
        return char_count // 4

    except Exception as e:
        logger.error(f"Error estimating tokens in {filing_path}: {e}")
        return 0


def analyze_filings_token_usage() -> Dict[str, Any]:
    """Analyze token usage across all filings.

    Returns:
        Dictionary with token usage statistics
    """
    cache_dir = Path(ETLConfig().filings_dir) / "cache"

    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return {}

    # Collect token usage data
    filing_tokens = []
    company_tokens = {}
    form_tokens = {}
    total_tokens = 0
    total_filings = 0

    # Scan all JSON files in the cache directory
    for file_path in cache_dir.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                filing_data = json.load(f)

            # Get metadata
            if "metadata" not in filing_data:
                continue

            metadata = filing_data["metadata"]
            company = metadata.get("ticker", "Unknown")
            form = metadata.get("form", "Unknown")

            # Estimate tokens
            tokens = estimate_tokens_in_filing(file_path)

            if tokens > 0:
                filing_tokens.append({"filing_id": file_path.stem, "company": company, "form": form, "tokens": tokens})

                # Update company stats
                if company not in company_tokens:
                    company_tokens[company] = 0
                company_tokens[company] += tokens

                # Update form stats
                if form not in form_tokens:
                    form_tokens[form] = 0
                form_tokens[form] += tokens

                total_tokens += tokens
                total_filings += 1

        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")

    # Calculate statistics
    token_counts = [f["tokens"] for f in filing_tokens]

    stats = {
        "total_tokens": total_tokens,
        "total_filings": total_filings,
        "avg_tokens_per_filing": total_tokens / max(1, total_filings),
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "median_tokens": np.median(token_counts) if token_counts else 0,
        "company_tokens": company_tokens,
        "form_tokens": form_tokens,
        "filing_tokens": filing_tokens,
    }

    return stats


def estimate_api_usage(stats: Dict[str, Any], batch_size: int = 50) -> Dict[str, Any]:
    """Estimate API usage based on token statistics.

    Args:
        stats: Token usage statistics
        batch_size: Batch size for embedding generation

    Returns:
        Dictionary with API usage estimates
    """
    total_tokens = stats["total_tokens"]
    total_filings = stats["total_filings"]
    avg_tokens = stats["avg_tokens_per_filing"]

    # Estimate number of API requests
    avg_chunks_per_filing = avg_tokens / batch_size
    total_chunks = total_tokens / batch_size

    # Estimate time to process all filings
    requests_per_minute = min(OPENAI_LIMITS["requests_per_minute"], OPENAI_LIMITS["tokens_per_minute"] / batch_size)
    minutes_to_process = total_chunks / requests_per_minute

    # Estimate daily capacity
    daily_token_capacity = OPENAI_LIMITS["tokens_per_day"]
    daily_request_capacity = OPENAI_LIMITS["requests_per_minute"] * 60 * 24
    daily_token_limit_filings = daily_token_capacity / avg_tokens
    daily_request_limit_filings = daily_request_capacity / avg_chunks_per_filing
    daily_filing_capacity = min(daily_token_limit_filings, daily_request_limit_filings)

    # Estimate optimal batch size
    # If we're token-limited, larger batches are better
    # If we're request-limited, smaller batches are better
    token_to_request_ratio = OPENAI_LIMITS["tokens_per_minute"] / OPENAI_LIMITS["requests_per_minute"]
    optimal_batch_size = token_to_request_ratio

    return {
        "total_api_requests": total_chunks,
        "avg_requests_per_filing": avg_chunks_per_filing,
        "minutes_to_process": minutes_to_process,
        "hours_to_process": minutes_to_process / 60,
        "daily_filing_capacity": daily_filing_capacity,
        "optimal_batch_size": optimal_batch_size,
        "limiting_factor": "tokens" if daily_token_limit_filings < daily_request_limit_filings else "requests",
    }


def generate_token_usage_report(stats: Dict[str, Any], api_usage: Dict[str, Any]) -> str:
    """Generate a report of token usage and API limits.

    Args:
        stats: Token usage statistics
        api_usage: API usage estimates

    Returns:
        Formatted report
    """
    report = "Token Usage and API Limits Report\n"
    report += "================================\n\n"

    # Token usage statistics
    report += "Token Usage Statistics:\n"
    report += f"  Total tokens: {stats['total_tokens']:,}\n"
    report += f"  Total filings: {stats['total_filings']}\n"
    report += f"  Average tokens per filing: {stats['avg_tokens_per_filing']:.2f}\n"
    report += f"  Minimum tokens: {stats['min_tokens']}\n"
    report += f"  Maximum tokens: {stats['max_tokens']}\n"
    report += f"  Median tokens: {stats['median_tokens']}\n\n"

    # API usage estimates
    report += "API Usage Estimates:\n"
    report += f"  Total API requests: {api_usage['total_api_requests']:.2f}\n"
    report += f"  Average requests per filing: {api_usage['avg_requests_per_filing']:.2f}\n"
    report += f"  Time to process all filings: {api_usage['hours_to_process']:.2f} hours\n"
    report += f"  Daily filing capacity: {api_usage['daily_filing_capacity']:.2f} filings\n"
    report += f"  Optimal batch size: {api_usage['optimal_batch_size']:.2f} tokens\n"
    report += f"  Limiting factor: {api_usage['limiting_factor']}\n\n"

    # OpenAI API limits
    report += "OpenAI API Limits:\n"
    report += f"  Tokens per minute (TPM): {OPENAI_LIMITS['tokens_per_minute']:,}\n"
    report += f"  Requests per minute (RPM): {OPENAI_LIMITS['requests_per_minute']:,}\n"
    report += f"  Tokens per day (TPD): {OPENAI_LIMITS['tokens_per_day']:,}\n\n"

    # Top companies by token usage
    report += "Top Companies by Token Usage:\n"
    top_companies = sorted(stats["company_tokens"].items(), key=lambda x: x[1], reverse=True)[:10]
    for company, tokens in top_companies:
        report += f"  {company}: {tokens:,} tokens\n"
    report += "\n"

    # Top forms by token usage
    report += "Top Forms by Token Usage:\n"
    top_forms = sorted(stats["form_tokens"].items(), key=lambda x: x[1], reverse=True)[:5]
    for form, tokens in top_forms:
        report += f"  {form}: {tokens:,} tokens\n"

    return report


def plot_token_distribution(stats: Dict[str, Any], output_dir: Path):
    """Plot token distribution across filings.

    Args:
        stats: Token usage statistics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get token counts
    token_counts = [f["tokens"] for f in stats["filing_tokens"]]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=50, alpha=0.7)
    plt.xlabel("Tokens per Filing")
    plt.ylabel("Number of Filings")
    plt.title("Distribution of Token Usage Across Filings")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "token_distribution.png")

    # Plot company token usage
    plt.figure(figsize=(12, 8))
    top_companies = sorted(stats["company_tokens"].items(), key=lambda x: x[1], reverse=True)[:15]
    companies = [c[0] for c in top_companies]
    tokens = [c[1] for c in top_companies]

    plt.barh(companies, tokens)
    plt.xlabel("Token Count")
    plt.ylabel("Company")
    plt.title("Top 15 Companies by Token Usage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "company_token_usage.png")

    # Plot form token usage
    plt.figure(figsize=(10, 6))
    forms = list(stats["form_tokens"].keys())
    form_tokens = list(stats["form_tokens"].values())

    plt.bar(forms, form_tokens)
    plt.xlabel("Form Type")
    plt.ylabel("Token Count")
    plt.title("Token Usage by Form Type")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "form_token_usage.png")


def main():
    """Main function to analyze token usage and API limits."""
    logger.info("Analyzing token usage across filings...")

    # Analyze token usage
    stats = analyze_filings_token_usage()

    if not stats:
        logger.error("No token usage statistics available.")
        return

    logger.info(f"Analyzed {stats['total_filings']} filings with {stats['total_tokens']:,} tokens")

    # Estimate API usage
    api_usage = estimate_api_usage(stats, batch_size=50)

    # Generate report
    report = generate_token_usage_report(stats, api_usage)

    # Save report
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "token_usage_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Token usage report saved to {report_path}")

    # Save raw data
    data_path = report_dir / "token_usage_data.json"
    with open(data_path, "w") as f:
        # Convert NumPy values to Python types for JSON serialization
        stats_copy = stats.copy()
        for key in ["avg_tokens_per_filing", "median_tokens"]:
            if key in stats_copy and isinstance(stats_copy[key], np.number):
                stats_copy[key] = float(stats_copy[key])

        json.dump({"stats": stats_copy, "api_usage": api_usage, "openai_limits": OPENAI_LIMITS}, f, indent=2)

    logger.info(f"Token usage data saved to {data_path}")

    # Generate plots
    try:
        plot_dir = report_dir / "plots"
        plot_token_distribution(stats, plot_dir)
        logger.info(f"Token usage plots saved to {plot_dir}")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")

    # Print summary
    print("\nToken Usage and API Limits Summary:")
    print(f"Total filings: {stats['total_filings']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Average tokens per filing: {stats['avg_tokens_per_filing']:.2f}")
    print(f"Time to process all filings: {api_usage['hours_to_process']:.2f} hours")
    print(f"Daily filing capacity: {api_usage['daily_filing_capacity']:.2f} filings")
    print(f"Optimal batch size: {api_usage['optimal_batch_size']:.2f} tokens")
    print(f"Limiting factor: {api_usage['limiting_factor']}")
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
