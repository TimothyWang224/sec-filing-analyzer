"""
Test Script for All Schema-Driven Tools

This script tests all tools that have been updated to use the schema-driven approach.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import (
    SchemaRegistry,
    SECDataTool,
    SECFinancialDataTool,
    SECGraphQueryTool,
    SECSemanticSearchTool,
    ToolDetailsTool,
    ToolRegistry,
)
from src.tools.schema_init import get_loaded_schemas, initialize_schemas


async def test_schema_registry():
    """Test the schema registry."""
    print("\n=== Testing Schema Registry ===")

    # Initialize schemas
    print("Initializing schemas...")
    success = initialize_schemas()
    print(f"Schema initialization {'successful' if success else 'failed'}")

    # List loaded schemas
    schemas = get_loaded_schemas()
    print(f"Loaded schemas: {schemas}")


async def test_tool_registry():
    """Test the tool registry."""
    print("\n=== Testing Tool Registry ===")

    # List all tools
    tools = ToolRegistry.list_tools()
    print(f"Registered tools: {list(tools.keys())}")

    # Get schema mappings for all tools
    for tool_name in tools.keys():
        mappings = ToolRegistry.get_schema_mappings(tool_name)
        if mappings:
            print(f"Schema mappings for {tool_name}: {mappings}")

    # Validate all schema mappings
    is_valid, all_errors = ToolRegistry.validate_all_schema_mappings()
    print(f"All schema mappings validation: {'valid' if is_valid else 'invalid'}")
    if all_errors:
        print(f"Validation errors: {all_errors}")


async def test_sec_data_tool():
    """Test the SEC data tool."""
    print("\n=== Testing SEC Data Tool ===")

    # Create the tool
    tool = SECDataTool()

    # Test parameter resolution
    original_params = {"ticker": "AAPL", "filing_type": "10-K", "start_date": "2022-01-01", "end_date": "2023-01-01"}

    print(f"Original parameters: {json.dumps(original_params, indent=2)}")

    # Resolve parameters
    resolved_params = tool.resolve_parameters(**original_params)
    print(f"Resolved parameters: {json.dumps(resolved_params, indent=2)}")

    # Execute the tool
    print("\nExecuting tool...")
    result = await tool.execute(**original_params)
    print(f"Tool result: {json.dumps(result, indent=2)}")


async def test_sec_semantic_search_tool():
    """Test the SEC semantic search tool."""
    print("\n=== Testing SEC Semantic Search Tool ===")

    # Create the tool
    try:
        tool = SECSemanticSearchTool(vector_store_path="data/test_vector_store")

        # Test parameter resolution
        original_params = {
            "query": "What are the risks related to COVID-19?",
            "companies": ["AAPL", "MSFT"],
            "top_k": 3,
            "filing_types": ["10-K", "10-Q"],
            "date_range": ["2020-01-01", "2023-01-01"],
        }

        print(f"Original parameters: {json.dumps(original_params, indent=2)}")

        # Resolve parameters
        resolved_params = tool.resolve_parameters(**original_params)
        print(f"Resolved parameters: {json.dumps(resolved_params, indent=2)}")

        # We'll skip actual execution since it requires a vector store
        print("\nSkipping execution as it requires a vector store")
    except Exception as e:
        print(f"Error testing SEC semantic search tool: {str(e)}")


async def test_sec_graph_query_tool():
    """Test the SEC graph query tool."""
    print("\n=== Testing SEC Graph Query Tool ===")

    # Create the tool
    try:
        tool = SECGraphQueryTool(use_neo4j=False)  # Use in-memory graph for testing

        # Test parameter resolution
        original_params = {
            "query_type": "company_filings",
            "parameters": {"ticker": "AAPL", "filing_types": ["10-K"], "limit": 5},
        }

        print(f"Original parameters: {json.dumps(original_params, indent=2)}")

        # Resolve parameters
        resolved_params = tool.resolve_parameters(**original_params)
        print(f"Resolved parameters: {json.dumps(resolved_params, indent=2)}")

        # We'll skip actual execution since it requires a graph database
        print("\nSkipping execution as it requires a graph database")
    except Exception as e:
        print(f"Error testing SEC graph query tool: {str(e)}")


async def test_sec_financial_data_tool():
    """Test the SEC financial data tool."""
    print("\n=== Testing SEC Financial Data Tool ===")

    # Create the tool
    try:
        tool = SECFinancialDataTool()

        # Test parameter resolution
        original_params = {
            "query_type": "financial_facts",
            "parameters": {
                "ticker": "AAPL",
                "metric": "Revenue",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "filing_type": "10-K",
            },
        }

        print(f"Original parameters: {json.dumps(original_params, indent=2)}")

        # Resolve parameters
        resolved_params = tool.resolve_parameters(**original_params)
        print(f"Resolved parameters: {json.dumps(resolved_params, indent=2)}")

        # Execute the tool
        print("\nExecuting tool...")
        result = await tool.execute(**original_params)
        print(f"Tool result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Error testing SEC financial data tool: {str(e)}")


async def test_tool_details_tool():
    """Test the tool details tool."""
    print("\n=== Testing Tool Details Tool ===")

    # Create the tool
    tool = ToolDetailsTool()

    # Execute the tool
    print("\nExecuting tool...")
    result = await tool.execute(tool_name="sec_data")
    print(f"Tool result: {json.dumps(result, indent=2)}")


async def main():
    """Main function."""
    print("Testing All Schema-Driven Tools")

    await test_schema_registry()
    await test_tool_registry()
    await test_sec_data_tool()
    await test_sec_semantic_search_tool()
    await test_sec_graph_query_tool()
    await test_sec_financial_data_tool()
    await test_tool_details_tool()


if __name__ == "__main__":
    asyncio.run(main())
