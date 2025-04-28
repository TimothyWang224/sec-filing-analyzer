"""
Tool registry for managing tools with automatic documentation extraction.
"""

import inspect
import logging
import re
from typing import Any, Dict, Optional, get_type_hints

from pydantic import Field

from ..contracts import ToolSpec
from .memoization import clear_tool_caches
from .schema_registry import SchemaRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for tools with automatic documentation extraction."""

    _tools: Dict[str, Dict[str, Any]] = {}
    _schema_mappings: Dict[str, Dict[str, str]] = {}
    _tool_specs: Dict[str, ToolSpec] = {}

    @classmethod
    def register(cls, tool_class=None, name=None, tags=None):
        """
        Register a tool with automatic documentation extraction.

        Can be used as a decorator:
        @ToolRegistry.register(tags=["sec"])
        class MyTool:
            ...

        Or called directly:
        ToolRegistry.register(MyTool, tags=["sec"])

        Args:
            tool_class: The tool class to register
            name: Optional name override (defaults to class name)
            tags: Optional tags for categorizing the tool

        Returns:
            The tool class (for decorator usage)
        """
        # Handle both decorator and direct call patterns
        if tool_class is None:
            # Used as a decorator with arguments
            def decorator(cls):
                return cls._register_tool(cls, name, tags)

            return decorator
        else:
            # Called directly
            return cls._register_tool(tool_class, name, tags)

    @classmethod
    def _register_tool(cls, tool_class, name=None, tags=None):
        """Internal method to register a tool."""
        # Extract name from class if not provided
        tool_name = name or tool_class.__name__.lower().replace("tool", "")

        # Check for schema information in the class
        db_schema = getattr(tool_class, "_db_schema", None)
        parameter_mappings = getattr(tool_class, "_parameter_mappings", None)

        # Register schema mappings if provided
        if db_schema and parameter_mappings:
            if tool_name not in cls._schema_mappings:
                cls._schema_mappings[tool_name] = {}

            cls._schema_mappings[tool_name].update(parameter_mappings)

            # Register mappings with the SchemaRegistry
            for param_name, field_name in parameter_mappings.items():
                SchemaRegistry.register_field_mapping(db_schema, param_name, field_name)

        # Extract documentation
        description = cls._extract_description(tool_class)
        parameters = cls._extract_parameters(tool_class)

        # Extract or generate compact description
        compact_description = getattr(tool_class, "_compact_description", None)
        if not compact_description:
            compact_description = description.split(".")[0]  # First sentence

        # Store in registry
        cls._tools[tool_name] = {
            "name": tool_name,
            "class": tool_class,
            "description": description,
            "compact_description": compact_description,
            "parameters": parameters,
            "tags": tags or [],
        }

        # Create and store a ToolSpec
        # For now, we'll use a dynamic BaseModel class for the input schema
        from pydantic import create_model

        # Create a dynamic model for each parameter
        input_models = {}
        for param_name, param_info in parameters.items():
            # Create field definitions for the model
            field_type = Any  # Default to Any
            field_info = {}

            # Add description
            field_info["description"] = param_info.get("description", f"Parameter {param_name}")

            # Add default if present
            if "default" in param_info and param_info["default"] is not None:
                field_info["default"] = param_info["default"]

            # Create the model
            param_model = create_model(
                f"{tool_name.capitalize()}{param_name.capitalize()}Params",
                **{param_name: (field_type, Field(**field_info))},
            )

            # Add to input models dictionary
            input_models[param_name] = param_model

        # Determine the output key based on the tool name
        output_key = tool_name

        # Create the ToolSpec
        cls._tool_specs[tool_name] = ToolSpec(
            name=tool_name,
            input_schema=input_models,
            output_key=output_key,
            description=description,
        )

        # Add metadata to the class
        tool_class._tool_name = tool_name
        tool_class._tool_tags = tags or []
        tool_class._compact_description = compact_description

        return tool_class

    @classmethod
    def _extract_description(cls, tool_class) -> str:
        """Extract description from class docstring."""
        docstring = tool_class.__doc__ or ""

        # Extract the first paragraph as the description
        description = docstring.strip().split("\n\n")[0].strip()

        return description or "No description available"

    @classmethod
    def _extract_parameters(cls, tool_class) -> Dict[str, Dict[str, Any]]:
        """Extract parameter documentation from execute method."""
        parameters = {}

        # Get the execute method
        if not hasattr(tool_class, "execute"):
            return parameters

        execute_method = tool_class.execute

        # Get signature of the execute method
        signature = inspect.signature(execute_method)

        # Get type hints
        try:
            type_hints = get_type_hints(execute_method)
        except Exception:
            type_hints = {}

        # Extract parameter info from signature
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            # Get type annotation
            param_type = "any"
            if param_name in type_hints:
                param_type = cls._format_type_hint(type_hints[param_name])
            elif param.annotation != inspect.Parameter.empty:
                param_type = cls._format_type_hint(param.annotation)

            # Check if parameter has default value
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default

            # Create parameter documentation
            parameters[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}",  # Default description
                "required": required,
            }

            if default is not None and default is not inspect.Parameter.empty:
                parameters[param_name]["default"] = default

        # Extract parameter descriptions from docstring
        docstring = execute_method.__doc__ or ""

        # Try to find Args section in docstring
        args_match = re.search(r"Args:(.*?)(?:Returns:|Raises:|$)", docstring, re.DOTALL)
        if args_match:
            args_section = args_match.group(1).strip()

            # Extract parameter descriptions
            param_matches = re.finditer(
                r"(\w+)(?:\s*\(\w+\))?\s*:\s*(.*?)(?=\n\s*\w+\s*:|$)",
                args_section,
                re.DOTALL,
            )

            for match in param_matches:
                param_name = match.group(1).strip()
                param_desc = match.group(2).strip()

                if param_name in parameters:
                    parameters[param_name]["description"] = param_desc

        return parameters

    @classmethod
    def _format_type_hint(cls, type_hint) -> str:
        """Format a type hint into a readable string."""
        type_str = str(type_hint)

        # Clean up typing module prefixes
        type_str = type_str.replace("typing.", "")

        # Handle common typing constructs
        type_str = type_str.replace("Optional[", "")
        type_str = type_str.replace("]", "")

        # Handle Union types
        if "Union[" in type_str:
            type_str = type_str.replace("Union[", "").replace("]", "")
            type_str = type_str.split(",")[0].strip()  # Take first type in union

        return type_str

    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool documentation by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            Tool documentation if found, None otherwise
        """
        # First, try to get the tool from the registry
        tool = cls._tools.get(name)

        # If the tool is not found, attempt a lazy import
        if tool is None:
            try:
                logger.info(f"Tool '{name}' not found in registry. Attempting lazy import.")

                # Try to import the tool module
                import importlib

                try:
                    # First try to import from sec_filing_analyzer.tools
                    importlib.import_module(f"sec_filing_analyzer.tools.{name}")
                    logger.info(f"Successfully imported sec_filing_analyzer.tools.{name}")
                except ImportError:
                    # Then try to import from src.tools
                    importlib.import_module(f"src.tools.{name}")
                    logger.info(f"Successfully imported src.tools.{name}")

                # Check if the tool is now registered
                tool = cls._tools.get(name)
                if tool:
                    logger.info(f"Tool '{name}' successfully registered via lazy import.")
                else:
                    logger.warning(f"Tool '{name}' module was imported but tool was not registered.")
            except Exception as e:
                logger.warning(f"Failed to lazy import tool '{name}': {str(e)}")

        return tool

    @classmethod
    def get_schema(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool input schema by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            Tool input schema if found, None otherwise
        """
        tool_spec = cls.get_tool_spec(name)
        if tool_spec:
            return tool_spec.input_schema
        return None

    @classmethod
    def get_tool_spec(cls, name: str) -> Optional[ToolSpec]:
        """
        Get tool specification by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            ToolSpec if found, None otherwise
        """
        # First, try to get the tool spec from the registry
        tool_spec = cls._tool_specs.get(name)

        # If the tool spec is not found, try to get the tool and create a spec
        if tool_spec is None:
            tool = cls.get(name)
            if tool:
                # Create a tool spec from the tool documentation
                from pydantic import create_model

                # Create a dynamic model for each parameter
                input_models = {}
                for param_name, param_info in tool["parameters"].items():
                    # Create field definitions for the model
                    field_type = Any  # Default to Any
                    field_info = {}

                    # Add description
                    field_info["description"] = param_info.get("description", f"Parameter {param_name}")

                    # Add default if present
                    if "default" in param_info and param_info["default"] is not None:
                        field_info["default"] = param_info["default"]

                    # Create the model
                    param_model = create_model(
                        f"{name.capitalize()}{param_name.capitalize()}Params",
                        **{param_name: (field_type, Field(**field_info))},
                    )

                    # Add to input models dictionary
                    input_models[param_name] = param_model

                # Determine the output key based on the tool name
                output_key = name

                # Create the ToolSpec
                tool_spec = ToolSpec(
                    name=name,
                    input_schema=input_models,
                    output_key=output_key,
                    description=tool["description"],
                )

                # Store the tool spec for future use
                cls._tool_specs[name] = tool_spec

        return tool_spec

    @classmethod
    def list_tools(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered tools.

        Returns:
            Dictionary mapping tool names to their documentation
        """
        return cls._tools.copy()

    @classmethod
    def get_schema_mappings(cls, tool_name: str) -> Dict[str, str]:
        """
        Get schema mappings for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dictionary mapping parameter names to field names
        """
        return cls._schema_mappings.get(tool_name, {})

    @classmethod
    def validate_schema_mappings(cls, tool_name: str) -> tuple[bool, list[str]]:
        """
        Validate schema mappings for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tuple of (is_valid, error_messages)
        """
        if tool_name not in cls._tools:
            return False, [f"Tool '{tool_name}' not found"]

        if tool_name not in cls._schema_mappings:
            return True, []  # No mappings to validate

        tool_class = cls._tools[tool_name]["class"]
        db_schema = getattr(tool_class, "_db_schema", None)

        if not db_schema:
            return False, [f"Tool '{tool_name}' has mappings but no database schema"]

        # Validate mappings against the schema
        errors = []
        for _, field_name in cls._schema_mappings[tool_name].items():
            field_info = SchemaRegistry.get_field_info(db_schema, field_name)
            if not field_info:
                errors.append(f"Field '{field_name}' not found in schema '{db_schema}'")

        return len(errors) == 0, errors

    @classmethod
    def validate_all_schema_mappings(cls) -> tuple[bool, dict[str, list[str]]]:
        """
        Validate all schema mappings.

        Returns:
            Tuple of (all_valid, {tool_name: error_messages})
        """
        all_valid = True
        all_errors = {}

        for tool_name in cls._schema_mappings:
            is_valid, errors = cls.validate_schema_mappings(tool_name)
            if not is_valid:
                all_valid = False
                all_errors[tool_name] = errors

        return all_valid, all_errors

    @classmethod
    def get_tool_documentation(cls, name: Optional[str] = None, format: str = "text") -> str:
        """
        Get formatted documentation for tools.

        Args:
            name: Optional name of specific tool to document
            format: Format of documentation ('text', 'markdown', or 'json')

        Returns:
            Formatted documentation string
        """
        if name:
            # Document specific tool
            tool_doc = cls.get(name)
            if not tool_doc:
                return f"Tool '{name}' not found"

            return cls._format_tool_doc(tool_doc, format)
        else:
            # Document all tools
            docs = []
            for tool_name in sorted(cls._tools.keys()):
                docs.append(cls._format_tool_doc(cls._tools[tool_name], format))

            if format == "json":
                import json

                return json.dumps(cls._tools, indent=2)
            else:
                return "\n\n".join(docs)

    @classmethod
    def clear_caches(cls):
        """
        Clear all tool caches.

        This should be called when data is updated to ensure tools use the latest data.
        """
        clear_tool_caches()
        logger.info("All tool caches cleared")

    @classmethod
    def get_compact_tool_documentation(cls, format: str = "text") -> str:
        """
        Get compact documentation for all tools.

        Args:
            format: Format of documentation ('text', 'markdown', or 'json')

        Returns:
            Formatted compact documentation string
        """
        if format == "json":
            compact_tools = {}
            for name, info in cls._tools.items():
                compact_tools[name] = {
                    "name": name,
                    "description": info.get("compact_description", ""),
                    "tags": info.get("tags", []),
                }
            import json

            return json.dumps(compact_tools, indent=2)

        elif format == "markdown":
            docs = []
            for name in sorted(cls._tools.keys()):
                info = cls._tools[name]
                description = info.get("compact_description", "")
                tags = info.get("tags", [])
                tag_str = f" *({', '.join(tags)})*" if tags else ""
                docs.append(f"- **{name}**{tag_str}: {description}")

            return "\n".join(docs)

        else:  # text
            docs = []
            for name in sorted(cls._tools.keys()):
                info = cls._tools[name]
                description = info.get("compact_description", "")
                docs.append(f"- {name}: {description}")

            return "\n".join(docs)

    @classmethod
    def _format_tool_doc(cls, tool_doc: Dict[str, Any], format: str) -> str:
        """Format a single tool's documentation."""
        name = tool_doc["name"]
        description = tool_doc["description"]
        parameters = tool_doc["parameters"]
        tags = tool_doc.get("tags", [])

        if format == "markdown":
            doc = f"## {name}\n\n"
            if tags:
                doc += f"*Tags: {', '.join(tags)}*\n\n"
            doc += f"{description}\n\n"

            if parameters:
                doc += "### Parameters\n\n"
                for param_name, param_info in parameters.items():
                    required = "**Required**" if param_info.get("required", False) else "Optional"
                    param_type = param_info.get("type", "any")
                    doc += f"- `{param_name}` ({param_type}) - {param_info.get('description', '')}"
                    doc += f" {required}.\n"

                    if "default" in param_info:
                        doc += f"  - Default: `{param_info['default']}`\n"

                    if "enum" in param_info:
                        doc += f"  - Allowed values: {', '.join([f'`{v}`' for v in param_info['enum']])}\n"

            return doc

        elif format == "text":
            doc = f"TOOL: {name}\n"
            if tags:
                doc += f"TAGS: {', '.join(tags)}\n"
            doc += f"DESCRIPTION: {description}\n"

            if parameters:
                doc += "PARAMETERS:\n"
                for param_name, param_info in parameters.items():
                    required = "Required" if param_info.get("required", False) else "Optional"
                    param_type = param_info.get("type", "any")
                    doc += f"  {param_name} ({param_type}): {param_info.get('description', '')}"
                    doc += f" {required}.\n"

                    if "default" in param_info:
                        doc += f"    Default: {param_info['default']}\n"

                    if "enum" in param_info:
                        doc += f"    Allowed values: {', '.join([str(v) for v in param_info['enum']])}\n"

            return doc

        else:  # json
            import json

            return json.dumps(tool_doc, indent=2)
