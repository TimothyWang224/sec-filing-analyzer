"""
Script to find all filings with zero vectors in the embeddings.

This script scans the filing cache directory and identifies any filings with zero vectors
in their embeddings, which indicates a failure in the embedding generation process.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sec_filing_analyzer.config import ETLConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def is_zero_vector(vector: List[float]) -> bool:
    """Check if a vector contains only zeros.

    Args:
        vector: The vector to check

    Returns:
        True if the vector contains only zeros, False otherwise
    """
    # Check the first 10 elements to determine if it's a zero vector
    # This is faster than checking the entire vector and still reliable
    return all(v == 0.0 for v in vector[:10])


def find_zero_vector_filings() -> List[Dict[str, Any]]:
    """Find all filings with zero vectors in their embeddings.

    Returns:
        List of dictionaries containing information about filings with zero vectors
    """
    cache_dir = Path(ETLConfig().filings_dir) / "cache"

    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return []

    zero_vector_filings = []

    # Scan all JSON files in the cache directory
    for file_path in cache_dir.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                filing_data = json.load(f)

            # Check if the filing has metadata
            if "metadata" not in filing_data:
                logger.warning(f"Missing metadata in filing: {file_path.name}")
                continue

            metadata = filing_data["metadata"]

            # Check if the filing has processed data
            if "processed_data" not in filing_data:
                logger.warning(f"Missing processed data in filing: {file_path.name}")
                continue

            processed_data = filing_data["processed_data"]

            # Check if the filing has an embedding
            if "embedding" not in processed_data:
                logger.warning(f"Missing embedding in filing: {file_path.name}")
                continue

            embedding = processed_data["embedding"]

            # Check if the embedding is a zero vector
            if is_zero_vector(embedding):
                zero_vector_filings.append(
                    {
                        "filing_id": file_path.stem,
                        "company": metadata.get("ticker", "Unknown"),
                        "form": metadata.get("form", "Unknown"),
                        "filing_date": metadata.get("filing_date", "Unknown"),
                        "file_path": str(file_path),
                    }
                )
                logger.info(f"Found zero vector in filing: {file_path.name}")

        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")

    return zero_vector_filings


def generate_report(zero_vector_filings: List[Dict[str, Any]]) -> str:
    """Generate a report of filings with zero vectors.

    Args:
        zero_vector_filings: List of filings with zero vectors

    Returns:
        A formatted report
    """
    if not zero_vector_filings:
        return "No filings with zero vectors found."

    # Group filings by company
    filings_by_company = {}
    for filing in zero_vector_filings:
        company = filing["company"]
        if company not in filings_by_company:
            filings_by_company[company] = []
        filings_by_company[company].append(filing)

    # Format the report
    report = "Zero Vector Filings Report\n"
    report += "========================\n\n"
    report += f"Total filings with zero vectors: {len(zero_vector_filings)}\n\n"

    for company, filings in filings_by_company.items():
        report += f"{company}: {len(filings)} filings\n"
        for filing in filings:
            report += f"  - {filing['form']} from {filing['filing_date']} (ID: {filing['filing_id']})\n"
        report += "\n"

    return report


def main():
    """Main function to find zero vector filings and generate a report."""
    logger.info("Scanning for filings with zero vectors...")

    zero_vector_filings = find_zero_vector_filings()

    report = generate_report(zero_vector_filings)
    print("\nZero Vector Filings Report:")
    print(report)

    # Save the report to a file
    report_path = Path("data/logs/zero_vector_filings_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Report saved to {report_path}")

    # Also save the raw data as JSON for further analysis
    json_path = Path("data/logs/zero_vector_filings.json")

    with open(json_path, "w") as f:
        json.dump(zero_vector_filings, f, indent=2)

    logger.info(f"Raw data saved to {json_path}")


if __name__ == "__main__":
    main()
