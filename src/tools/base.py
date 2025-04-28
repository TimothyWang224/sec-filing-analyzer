import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional

from .memoization import memoize_tool
from .registry import ToolRegistry
from .schema_registry import SchemaRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the ConfigProvider
try:
    from sec_filing_analyzer.config import ConfigProvider

    HAS_CONFIG_PROVIDER = True
except ImportError:
    HAS_CONFIG_PROVIDER = False
    logger.warning(
        "Could not import ConfigProvider. Tool schemas will be loaded from registry only."
    )

# Re-export the register_tool decorator from ToolRegistry
register_tool = ToolRegistry.register


class Tool(ABC):
    """
    Base class for all tools available to agents.

    All tools should inherit from this class and implement the _execute method.
    Tools provide standardized error handling and response formatting through
    the format_error_response and format_success_response methods.

    All tool responses will have the following standard fields:
    - query_type: The type of query that was executed
    - parameters: The parameters that were used
    - results: The results of the query (empty list for errors)
    - output_key: The tool's name
    - success: Boolean indicating whether the operation was successful

    Error responses will additionally have:
    - error or warning: The error message (depending on error_type)

    Tools may add additional fields to the response as needed.
    """

    # Class variables for tool metadata
    _tool_name: ClassVar[Optional[str]] = None
    _tool_tags: ClassVar[Optional[list]] = None
    _compact_description: ClassVar[Optional[str]] = (
        None  # New class variable for compact description
    )
    _db_schema: ClassVar[Optional[str]] = (
        None  # Database schema name for parameter resolution
    )
    _parameter_mappings: ClassVar[Optional[Dict[str, str]]] = (
        None  # Parameter to field mappings
    )

    def __init__(self) -> None:
        """Initialize a tool."""
        # Get name from class if registered, otherwise use class name
        self.name = getattr(
            self.__class__,
            "_tool_name",
            self.__class__.__name__.lower().replace("tool", ""),
        )

        # Get description from class docstring
        self.description = (
            self.__class__.__doc__.strip().split("\n\n")[0].strip()
            if self.__class__.__doc__
            else "No description available"
        )

        # Get compact description or generate one from the full description
        self.compact_description = getattr(
            self.__class__, "_compact_description", self.description.split(".")[0]
        )

        # Get tags from class if registered
        self.tags = getattr(self.__class__, "_tool_tags", [])

    @abstractmethod
    async def _execute_abstract(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Abstract execute method to be implemented by subclasses.

        Args:
            **kwargs: Keyword arguments based on the tool's parameters

        Returns:
            The result of the tool's execution
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get the tool's metadata."""
        return {"name": self.name, "description": self.description, "tags": self.tags}

    @classmethod
    def get_documentation(cls, format: str = "text") -> str:
        """Get the tool's documentation."""
        tool_name = getattr(cls, "_tool_name", cls.__name__.lower().replace("tool", ""))
        return ToolRegistry.get_tool_documentation(tool_name, format)

    def validate_args(self, **kwargs: Any) -> bool:
        """
        Validate the arguments before execution.

        Args:
            **kwargs: Keyword arguments to validate

        Returns:
            True if arguments are valid, False otherwise
        """
        # Try to get parameter info from ConfigProvider first
        parameters: Dict[str, Any] = {}
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

    def resolve_parameters(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Resolve dynamic parameters to their database field names.

        Args:
            **kwargs: Keyword arguments to resolve

        Returns:
            Dictionary with resolved parameters
        """
        # If no database schema is defined, return the original parameters
        db_schema = getattr(self.__class__, "_db_schema", None)
        if not db_schema:
            return kwargs

        # Try to get parameter mappings from ConfigProvider first
        param_mappings: Dict[str, str] = {}
        if HAS_CONFIG_PROVIDER:
            try:
                # Get schema registry from ConfigProvider
                schema_registry = ConfigProvider.get_config(SchemaRegistry)
                if schema_registry:
                    # Get mappings for this schema
                    mappings = schema_registry.get_field_mappings(db_schema)
                    if mappings:
                        param_mappings = mappings
                        logger.debug(
                            f"Using parameter mappings from ConfigProvider for {db_schema}"
                        )
            except Exception as e:
                logger.warning(
                    f"Error getting parameter mappings from ConfigProvider: {str(e)}"
                )

        # Fall back to class attribute if no mappings were found
        if not param_mappings:
            class_mappings = getattr(self.__class__, "_parameter_mappings", {})
            if class_mappings:
                param_mappings = class_mappings
            else:
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

    # This method is already defined above as an abstract method

    def format_error_response(
        self,
        query_type: str,
        parameters: Dict[str, Any],
        error_message: str,
        error_type: str = "error",
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format a standardized error response.

        Args:
            query_type: The type of query that was attempted
            parameters: The parameters that were used
            error_message: The error message
            error_type: The type of error (default: "error")
            additional_data: Additional data to include in the response

        Returns:
            A standardized error response dictionary
        """
        response = {
            "query_type": query_type,
            "parameters": parameters,
            error_type: error_message,
            "results": [],
            "output_key": self.name,
            "success": False,
        }

        # Add any additional data
        if additional_data:
            response.update(additional_data)

        return response

    def format_success_response(
        self,
        query_type: str,
        parameters: Dict[str, Any],
        results: Any,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format a standardized success response.

        Args:
            query_type: The type of query that was executed
            parameters: The parameters that were used
            results: The results of the query
            additional_data: Additional data to include in the response

        Returns:
            A standardized success response dictionary
        """
        response = {
            "query_type": query_type,
            "parameters": parameters,
            "results": results,
            "output_key": self.name,
            "success": True,
        }

        # Add any additional data
        if additional_data:
            response.update(additional_data)

        return response

    @memoize_tool
    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the tool with the given arguments.

        This method handles parameter validation and resolution before
        delegating to the _execute method.

        Args:
            **kwargs: Keyword arguments based on the tool's parameters

        Returns:
            A standardized response dictionary with the following fields:
            - query_type: The type of query that was executed
            - parameters: The parameters that were used
            - results: The results of the query (empty list for errors)
            - output_key: The tool's name
            - success: Boolean indicating whether the operation was successful

            Error responses will additionally have:
            - error or warning: The error message (depending on error_type)

            Tools may add additional fields to the response as needed.
        """
        # Validate arguments
        self.validate_args(**kwargs)

        # Resolve parameters
        resolved_params = self.resolve_parameters(**kwargs)

        # Log the parameter resolution if there were changes
        if resolved_params != kwargs:
            logger.info(
                f"Resolved parameters for {self.name}: {kwargs} -> {resolved_params}"
            )

        # Execute the tool with resolved parameters
        result = await self._execute_abstract(**resolved_params)

        # Get the tool spec to determine the output key
        from .registry import ToolRegistry

        tool_spec = ToolRegistry.get_tool_spec(self.name)

        # If we have a tool spec, add the output_key to the result
        if tool_spec and isinstance(result, dict):
            # If the result doesn't already have an output_key field, add it
            if "output_key" not in result:
                result["output_key"] = tool_spec.output_key

        return result
