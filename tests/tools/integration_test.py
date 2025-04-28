"""
Integration tests for the SEC tools.
"""

import asyncio
import logging
from unittest.mock import patch

from src.tools.sec_data import SECDataTool
from src.tools.sec_financial_data import SECFinancialDataTool
from src.tools.sec_graph_query import SECGraphQueryTool
from src.tools.sec_semantic_search import SECSemanticSearchTool
from src.tools.tool_details import ToolDetailsTool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_sec_data_tool():
    """Test the SECDataTool."""
    logger.info("Testing SECDataTool...")

    # Create the tool
    tool = SECDataTool()

    # Execute the tool
    result = await tool.execute(
        query_type="sec_data",
        parameters={
            "ticker": "AAPL",
            "filing_type": "10-K",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
        },
    )

    # Log the result
    logger.info(f"SECDataTool result: {result}")

    return result


async def test_sec_financial_data_tool():
    """Test the SECFinancialDataTool."""
    logger.info("Testing SECFinancialDataTool...")

    # Create the tool
    tool = SECFinancialDataTool()

    # Execute the tool
    result = await tool.execute(
        query_type="financial_facts",
        parameters={
            "ticker": "AAPL",
            "metrics": ["Revenue"],
            "start_date": "2022-01-01",
            "end_date": "2022-12-31",
        },
    )

    # Log the result
    logger.info(f"SECFinancialDataTool result: {result}")

    return result


async def test_sec_semantic_search_tool():
    """Test the SECSemanticSearchTool."""
    logger.info("Testing SECSemanticSearchTool...")

    # Create the tool
    tool = SECSemanticSearchTool()

    # Execute the tool
    result = await tool.execute(
        query_type="semantic_search",
        parameters={
            "query": "What are Apple's risks related to supply chain?",
            "companies": ["AAPL"],
            "top_k": 3,
            "filing_types": ["10-K"],
        },
    )

    # Log the result
    logger.info(f"SECSemanticSearchTool result: {result}")

    return result


async def test_sec_graph_query_tool():
    """Test the SECGraphQueryTool."""
    logger.info("Testing SECGraphQueryTool...")

    # Create a mock GraphStore class that can handle the two-parameter call
    class MockGraphStore:
        def __init__(self, *args, **kwargs):
            pass

        def query(self, query, parameters=None):
            # Just return some mock data
            return [
                {
                    "filing_type": "10-K",
                    "filing_date": "2022-10-28",
                    "accession_number": "0000320193-22-000108",
                    "fiscal_year": "2022",
                    "fiscal_period": "FY",
                }
            ]

    # Create the tool with our mock
    with patch("src.tools.sec_graph_query.GraphStore", MockGraphStore):
        tool = SECGraphQueryTool()

        # Execute the tool
        result = await tool.execute(
            query_type="company_filings",
            parameters={"ticker": "AAPL", "filing_types": ["10-K"], "limit": 5},
        )

        # Log the result
        logger.info(f"SECGraphQueryTool result: {result}")

        return result


async def test_tool_details_tool():
    """Test the ToolDetailsTool."""
    logger.info("Testing ToolDetailsTool...")

    # Create the tool
    tool = ToolDetailsTool()

    # Execute the tool
    result = await tool.execute(
        query_type="tool_details", parameters={"tool_name": "sec_data"}
    )

    # Log the result
    logger.info(f"ToolDetailsTool result: {result}")

    return result


async def main():
    """Run all integration tests."""
    try:
        # Test SECDataTool
        await test_sec_data_tool()

        # Test SECFinancialDataTool
        await test_sec_financial_data_tool()

        # Test SECSemanticSearchTool
        await test_sec_semantic_search_tool()

        # Test SECGraphQueryTool
        await test_sec_graph_query_tool()

        # Test ToolDetailsTool
        await test_tool_details_tool()

    except Exception as e:
        logger.error(f"Error running integration tests: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
