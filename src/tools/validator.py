"""
Tool call validation utilities.

This module provides utilities for validating tool calls before execution,
ensuring that parameters are valid and that the tool can handle the request.
"""

from typing import Any, Dict, Type

from pydantic import ValidationError

from ..contracts import BaseModel, ToolInput
from ..errors import ParameterError, QueryTypeUnsupported
from .registry import ToolRegistry


def validate_call(tool_name: str, query_type: str, params: Dict[str, Any]) -> None:
    """
    Validate a tool call before execution.

    Args:
        tool_name: The name of the tool to validate for
        query_type: The type of query to validate
        params: The parameters to validate

    Raises:
        QueryTypeUnsupported: If the query type is not supported by the tool
        ParameterError: If the parameters are invalid
    """
    # Get the tool spec
    tool_spec = ToolRegistry.get_tool_spec(tool_name)
    if not tool_spec:
        raise ValueError(f"Tool '{tool_name}' not found in registry")

    # Check if the query type is supported
    if query_type not in tool_spec.input_schema:
        supported_types = list(tool_spec.input_schema.keys())
        raise QueryTypeUnsupported(query_type, tool_name, supported_types)

    # Get the parameter model
    param_model: Type[BaseModel] = tool_spec.input_schema[query_type]

    # Validate the parameters
    try:
        param_model(**params)
    except ValidationError as e:
        # Extract field information from the validation error
        field = None
        if e.errors() and "loc" in e.errors()[0]:
            field = e.errors()[0]["loc"][0]

        raise ParameterError(str(e), {"field": field})


def validate_tool_input(tool_name: str, tool_input: ToolInput) -> None:
    """
    Validate a ToolInput object before execution.

    Args:
        tool_name: The name of the tool to validate for
        tool_input: The ToolInput object to validate

    Raises:
        QueryTypeUnsupported: If the query type is not supported by the tool
        ParameterError: If the parameters are invalid
    """
    validate_call(tool_name, tool_input.query_type, tool_input.parameters)


def create_tool_input(query_type: str, parameters: Dict[str, Any]) -> ToolInput:
    """
    Create a ToolInput object from a query type and parameters.

    Args:
        query_type: The type of query
        parameters: The parameters for the query

    Returns:
        A ToolInput object
    """
    return ToolInput(query_type=query_type, parameters=parameters)
