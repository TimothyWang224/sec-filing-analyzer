"""
Test Script for Schema-Driven Tools

This script demonstrates the schema-driven approach to tool implementation.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import SchemaRegistry, SECFinancialDataTool, ToolRegistry
from src.tools.schema_init import get_loaded_schemas, get_schema_fields, initialize_schemas


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

    # Get fields for each schema
    for schema_name in schemas:
        fields = get_schema_fields(schema_name)
        print(f"Fields in {schema_name}: {fields}")

        # Get field details for a few fields
        for field_name in fields[:3]:  # Just show the first 3 fields
            field_info = SchemaRegistry.get_field_info(schema_name, field_name)
            print(f"  {field_name}: {field_info}")

            # Get field aliases
            aliases = SchemaRegistry.get_field_aliases(schema_name, field_name)
            print(f"  Aliases for {field_name}: {aliases}")

    # Test field resolution
    print("\nTesting field resolution:")
    for schema_name in schemas:
        # Try to resolve a field by alias
        if schema_name == "financial_facts":
            field = SchemaRegistry.find_field_by_alias(schema_name, "metric")
            print(f"Resolved 'metric' to '{field}' in {schema_name}")

            field = SchemaRegistry.find_field_by_alias(schema_name, "date")
            print(f"Resolved 'date' to '{field}' in {schema_name}")


async def test_tool_registry():
    """Test the tool registry."""
    print("\n=== Testing Tool Registry ===")

    # List all tools
    tools = ToolRegistry.list_tools()
    print(f"Registered tools: {list(tools.keys())}")

    # Get schema mappings for a tool
    mappings = ToolRegistry.get_schema_mappings("sec_financial_data")
    print(f"Schema mappings for sec_financial_data: {mappings}")

    # Validate schema mappings
    is_valid, errors = ToolRegistry.validate_schema_mappings("sec_financial_data")
    print(f"Schema mappings validation for sec_financial_data: {'valid' if is_valid else 'invalid'}")
    if errors:
        print(f"Validation errors: {errors}")

    # Validate all schema mappings
    is_valid, all_errors = ToolRegistry.validate_all_schema_mappings()
    print(f"All schema mappings validation: {'valid' if is_valid else 'invalid'}")
    if all_errors:
        print(f"Validation errors: {all_errors}")


async def test_sec_financial_data_tool():
    """Test the SEC financial data tool."""
    print("\n=== Testing SEC Financial Data Tool ===")

    # Create the tool
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


async def main():
    """Main function."""
    print("Testing Schema-Driven Tools")

    await test_schema_registry()
    await test_tool_registry()
    await test_sec_financial_data_tool()


if __name__ == "__main__":
    asyncio.run(main())
