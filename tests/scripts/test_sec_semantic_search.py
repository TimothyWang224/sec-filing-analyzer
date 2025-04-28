"""
Test script for the SEC Semantic Search Tool.

This script tests the functionality of the SECSemanticSearchTool by performing
various semantic searches on SEC filings.
"""

import argparse
import asyncio
import logging
from typing import List, Optional

from src.tools.sec_semantic_search import SECSemanticSearchTool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_semantic_search(
    query: str,
    companies: Optional[List[str]] = None,
    top_k: int = 5,
    filing_types: Optional[List[str]] = None,
    date_range: Optional[List[str]] = None,
    sections: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    hybrid_search_weight: float = 0.5,
):
    """Test the semantic search tool with the given parameters."""
    try:
        # Initialize the tool
        tool = SECSemanticSearchTool()

        # Convert date_range to tuple if provided
        date_range_tuple = tuple(date_range) if date_range else None

        # Execute the search
        logger.info(f"Executing semantic search: {query}")
        results = await tool.execute(
            query=query,
            companies=companies,
            top_k=top_k,
            filing_types=filing_types,
            date_range=date_range_tuple,
            sections=sections,
            keywords=keywords,
            hybrid_search_weight=hybrid_search_weight,
        )

        # Print results
        print("\n=== Semantic Search Results ===")
        print(f"Query: {query}")
        print(f"Companies: {companies}")
        print(f"Filing Types: {filing_types}")
        print(f"Date Range: {date_range}")
        print(f"Total Results: {results['total_results']}")
        print("\nTop Results:")

        for i, result in enumerate(results["results"]):
            print(f"\n--- Result {i + 1} ---")
            print(f"Score: {result['score']:.4f}")
            print(
                f"Company: {result['metadata']['company']} ({result['metadata']['ticker']})"
            )
            print(
                f"Filing: {result['metadata']['filing_type']} ({result['metadata']['filing_date']})"
            )
            print(
                f"Section: {result['metadata']['section']} ({result['metadata']['section_type']})"
            )
            print(f"Text: {result['text'][:200]}...")

        return results

    except Exception as e:
        logger.error(f"Error testing semantic search: {str(e)}")
        raise


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test the SEC Semantic Search Tool")
    parser.add_argument(
        "--query",
        type=str,
        default="revenue growth and profitability",
        help="Search query text",
    )
    parser.add_argument(
        "--companies",
        type=str,
        nargs="*",
        help="List of company tickers to search within",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of results to return"
    )
    parser.add_argument(
        "--filing_types", type=str, nargs="*", help="List of filing types to filter by"
    )
    parser.add_argument(
        "--start_date", type=str, help="Start date for filing search (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date", type=str, help="End date for filing search (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--sections", type=str, nargs="*", help="List of document sections to filter by"
    )
    parser.add_argument(
        "--keywords", type=str, nargs="*", help="List of keywords to search for"
    )
    parser.add_argument(
        "--hybrid_weight",
        type=float,
        default=0.5,
        help="Weight for hybrid search (0.0 = pure vector, 1.0 = pure keyword)",
    )

    args = parser.parse_args()

    # Prepare date range if both start and end dates are provided
    date_range = None
    if args.start_date and args.end_date:
        date_range = [args.start_date, args.end_date]

    # Run the test
    asyncio.run(
        test_semantic_search(
            query=args.query,
            companies=args.companies,
            top_k=args.top_k,
            filing_types=args.filing_types,
            date_range=date_range,
            sections=args.sections,
            keywords=args.keywords,
            hybrid_search_weight=args.hybrid_weight,
        )
    )


if __name__ == "__main__":
    main()
