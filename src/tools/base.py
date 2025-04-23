from typing import Any, Dict, ClassVar, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
import logging
from .registry import ToolRegistry
from .schema_registry import SchemaRegistry
from .memoization import memoize_tool, clear_tool_caches

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the ConfigProvider
try:
    from sec_filing_analyzer.config import ConfigProvider
    HAS_CONFIG_PROVIDER = True
except ImportError:
    HAS_CONFIG_PROVIDER = False
    logger.warning("Could not import ConfigProvider. Tool schemas will be loaded from registry only.")

# Re-export the register_tool decorator from ToolRegistry
register_tool = ToolRegistry.register

class Tool(ABC):
    """Base class for all tools available to agents."""

    # Class variables for tool metadata
    _tool_name: ClassVar[str] = None
    _tool_tags: ClassVar[list] = None
    _compact_description: ClassVar[str] = None  # New class variable for compact description
    _db_schema: ClassVar[str] = None  # Database schema name for parameter resolution
    _parameter_mappings: ClassVar[Dict[str, str]] = None  # Parameter to field mappings

    def __init__(self):
        """Initialize a tool."""
        # Get name from class if registered, otherwise use class name
        self.name = getattr(self.__class__, '_tool_name', self.__class__.__name__.lower().replace('tool', ''))

        # Get description from class docstring
        self.description = self.__class__.__doc__.strip().split('\n\n')[0].strip() if self.__class__.__doc__ else "No description available"

        # Get compact description or generate one from the full description
        self.compact_description = getattr(self.__class__, '_compact_description', self.description.split('.')[0])

        # Get tags from class if registered
        self.tags = getattr(self.__class__, '_tool_tags', [])

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Keyword arguments based on the tool's parameters

        Returns:
            The result of the tool's execution
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get the tool's metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags
        }

    @classmethod
    def get_documentation(cls, format: str = "text") -> str:
        """Get the tool's documentation."""
        tool_name = getattr(cls, '_tool_name', cls.__name__.lower().replace('tool', ''))
        return ToolRegistry.get_tool_documentation(tool_name, format)

    def validate_args(self, **kwargs) -> bool:
        """
        Validate the arguments before execution.

        Args:
            **kwargs: Keyword arguments to validate

        Returns:
            True if arguments are valid, False otherwise
        """
        # Try to get parameter info from ConfigProvider first
        parameters = {}
        if HAS_CONFIG_PROVIDER:
            try:
                # Get schema from ConfigProvider
                schema = ConfigProvider.get_tool_schema(self.name)
                if schema:
                    parameters = schema
                    logger.debug(f"Using schema from ConfigProvider for {self.name}")
            except Exception as e:
                logger.warning(f"Error getting schema from ConfigProvider: {str(e)}")

        # Fall back to registry if no schema was found
        if not parameters:
            tool_info = ToolRegistry.get(self.name)
            if not tool_info:
                return True  # Can't validate if not registered

            parameters = tool_info.get("parameters", {})

        # Check required parameters
        for param_name, param_info in parameters.items():
            if param_info.get("required", False) and param_name not in kwargs:
                raise ValueError(f"Missing required parameter: {param_name}")

        return True

    def resolve_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Resolve dynamic parameters to their database field names.

        Args:
            **kwargs: Keyword arguments to resolve

        Returns:
            Dictionary with resolved parameters
        """
        # If no database schema is defined, return the original parameters
        db_schema = getattr(self.__class__, '_db_schema', None)
        if not db_schema:
            return kwargs

        # Try to get parameter mappings from ConfigProvider first
        param_mappings = {}
        if HAS_CONFIG_PROVIDER:
            try:
                # Get schema registry from ConfigProvider
                schema_registry = ConfigProvider.get_config(SchemaRegistry)
                if schema_registry:
                    # Get mappings for this schema
                    param_mappings = schema_registry.get_field_mappings(db_schema)
                    if param_mappings:
                        logger.debug(f"Using parameter mappings from ConfigProvider for {db_schema}")
            except Exception as e:
                logger.warning(f"Error getting parameter mappings from ConfigProvider: {str(e)}")

        # Fall back to class attribute if no mappings were found
        if not param_mappings:
            param_mappings = getattr(self.__class__, '_parameter_mappings', {})
            if not param_mappings:
                return kwargs

        # Resolve parameters
        resolved_params = kwargs.copy()
        for param_name, param_value in kwargs.items():
            if param_name in param_mappings:
                field_name = param_mappings[param_name]
                # Replace the parameter with the field name in the resolved parameters
                resolved_params[field_name] = param_value
                # Remove the original parameter if it's different from the field name
                if field_name != param_name and param_name in resolved_params:
                    del resolved_params[param_name]

        return resolved_params

    async def _execute(self, **kwargs) -> Any:
        """
        Internal execute method to be implemented by subclasses.

        Args:
            **kwargs: Keyword arguments for the tool

        Returns:
            The result of the tool's execution
        """
        raise NotImplementedError("Subclasses must implement _execute")

    @memoize_tool
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with the given arguments.

        This method handles parameter validation and resolution before
        delegating to the _execute method.

        Args:
            **kwargs: Keyword arguments based on the tool's parameters

        Returns:
            The result of the tool's execution with output_key information
        """
        # Validate arguments
        self.validate_args(**kwargs)

        # Resolve parameters
        resolved_params = self.resolve_parameters(**kwargs)

        # Log the parameter resolution if there were changes
        if resolved_params != kwargs:
            logger.info(f"Resolved parameters for {self.name}: {kwargs} -> {resolved_params}")

        # Execute the tool with resolved parameters
        result = await self._execute(**resolved_params)

        # Get the tool spec to determine the output key
        from .registry import ToolRegistry
        tool_spec = ToolRegistry.get_tool_spec(self.name)

        # If we have a tool spec, add the output_key to the result
        if tool_spec and isinstance(result, dict):
            # If the result doesn't already have an output_key field, add it
            if "output_key" not in result:
                result["output_key"] = tool_spec.output_key

        return result