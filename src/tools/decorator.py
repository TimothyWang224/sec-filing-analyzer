"""
Tool Decorator

This module provides a decorator for registering tools with the ToolRegistry
and configuring their schema mappings.
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .base import Tool
from .registry import ToolRegistry
from .schema_registry import SchemaRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tool(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    db_schema: Optional[str] = None,
    parameter_mappings: Optional[Dict[str, str]] = None,
    compact_description: Optional[str] = None,
):
    """
    Decorator for registering a tool with the ToolRegistry and configuring schema mappings.

    Args:
        name: Optional name override (defaults to class name)
        tags: Optional tags for categorizing the tool
        db_schema: Optional database schema name for parameter resolution
        parameter_mappings: Optional parameter to field mappings
        compact_description: Optional compact description for the tool

    Returns:
        Decorated tool class
    """

    def decorator(cls):
        # Set class variables
        if name:
            cls._tool_name = name
        if tags:
            cls._tool_tags = tags
        if db_schema:
            cls._db_schema = db_schema
        if parameter_mappings:
            cls._parameter_mappings = parameter_mappings
        if compact_description:
            cls._compact_description = compact_description

        # Register the tool with the ToolRegistry
        tool_name = name or cls.__name__.lower().replace("tool", "")
        ToolRegistry._register_tool(cls, name=tool_name, tags=tags)

        # Register parameter mappings with the SchemaRegistry if a schema is provided
        if db_schema and parameter_mappings:
            for param_name, field_name in parameter_mappings.items():
                SchemaRegistry.register_field_mapping(db_schema, param_name, field_name)

        # Ensure the class implements _execute
        original_execute = getattr(cls, "execute", None)

        # If the class doesn't have an _execute method but has an execute method,
        # create an _execute method that calls the execute method
        if not hasattr(cls, "_execute") and original_execute:

            @wraps(original_execute)
            async def _execute(self, **kwargs):
                # Call the original execute method
                return await original_execute(self, **kwargs)

            # Add the _execute method to the class
            cls._execute = _execute

            # Log a warning about the deprecated pattern
            logger.warning(
                f"Tool {cls.__name__} uses deprecated 'execute' method. Please implement '_execute' instead."
            )

        return cls

    return decorator
