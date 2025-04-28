#!/usr/bin/env python
"""
Test the compact tool descriptions and two-stage tool selection.

This script tests the new compact tool descriptions and the two-stage
tool selection process with the tool_details tool.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agents.financial_analyst import FinancialAnalystAgent
from src.environments.base import Environment
from src.tools.registry import ToolRegistry


async def test_compact_tool_documentation():
    """Test the compact tool documentation."""
    # Get compact tool documentation
    compact_docs = ToolRegistry.get_compact_tool_documentation(format="text")
    print("\n=== Compact Tool Documentation (Text) ===")
    print(compact_docs)

    # Get compact tool documentation in markdown format
    compact_docs_md = ToolRegistry.get_compact_tool_documentation(format="markdown")
    print("\n=== Compact Tool Documentation (Markdown) ===")
    print(compact_docs_md)

    # Get compact tool documentation in JSON format
    compact_docs_json = ToolRegistry.get_compact_tool_documentation(format="json")
    print("\n=== Compact Tool Documentation (JSON) ===")
    print(compact_docs_json)


async def test_tool_details_tool():
    """Test the tool details tool."""
    # Create an environment
    env = Environment()

    # Test the tool_details tool
    print("\n=== Tool Details Tool ===")
    result = await env.execute_action({"tool": "tool_details", "args": {"tool_name": "sec_semantic_search"}})

    print(json.dumps(result, indent=2))


async def test_two_stage_tool_selection():
    """Test the two-stage tool selection process."""
    # Create a financial analyst agent
    agent = FinancialAnalystAgent()

    # Test tool selection with a simple query
    print("\n=== Two-Stage Tool Selection ===")
    print("Query: Analyze Apple's financial performance")

    tool_calls = await agent.select_tools("Analyze Apple's financial performance")

    print("\nSelected Tools:")
    print(json.dumps(tool_calls, indent=2))


async def main():
    """Run all tests."""
    await test_compact_tool_documentation()
    await test_tool_details_tool()
    await test_two_stage_tool_selection()


if __name__ == "__main__":
    asyncio.run(main())
