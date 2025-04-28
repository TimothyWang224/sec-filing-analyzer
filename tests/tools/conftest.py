"""
Pytest fixtures for the tools tests.
"""

import pytest

from src.tools.registry import ToolRegistry
from src.tools.schema_registry import SchemaRegistry


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset the tool and schema registries before each test."""
    # Clear the tool registry
    ToolRegistry._tools = {}
    ToolRegistry._schema_mappings = {}
    ToolRegistry._tool_specs = {}

    # Clear the schema registry
    SchemaRegistry._db_schemas = {}
    SchemaRegistry._field_mappings = {}
    SchemaRegistry._schema_files = {}

    yield
