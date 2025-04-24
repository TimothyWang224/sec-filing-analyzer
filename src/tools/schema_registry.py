"""
Database Schema Registry

This module provides a registry for database schemas and field mappings.
It serves as a central repository for database field definitions and
their mappings to tool parameters.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaRegistry:
    """Registry for database schemas and field mappings."""

    # Class variables to store schemas and mappings
    _db_schemas: Dict[str, Dict[str, Any]] = {}
    _field_mappings: Dict[str, Dict[str, str]] = {}
    _schema_files: Dict[str, str] = {}

    @classmethod
    def load_schema(cls, schema_name: str, schema_file: str) -> bool:
        """
        Load a database schema from a JSON file.

        Args:
            schema_name: Name of the schema (e.g., 'financial_facts')
            schema_file: Path to the schema JSON file

        Returns:
            True if schema was loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(schema_file):
                logger.error(f"Schema file not found: {schema_file}")
                return False

            with open(schema_file, 'r') as f:
                schema = json.load(f)

            cls._db_schemas[schema_name] = schema
            cls._schema_files[schema_name] = schema_file
            logger.info(f"Loaded schema '{schema_name}' from {schema_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading schema '{schema_name}' from {schema_file}: {str(e)}")
            return False

    @classmethod
    def register_field_mapping(cls, schema_name: str, param_name: str, field_name: str) -> None:
        """
        Register a mapping between a tool parameter and a database field.

        Args:
            schema_name: Name of the schema
            param_name: Name of the tool parameter
            field_name: Name of the database field
        """
        # Initialize the schema mapping dictionary if it doesn't exist
        if schema_name not in cls._field_mappings:
            cls._field_mappings[schema_name] = {}

        # Check if this mapping already exists
        if param_name in cls._field_mappings[schema_name]:
            existing_field = cls._field_mappings[schema_name][param_name]
            # If the mapping is the same, just return without logging
            if existing_field == field_name:
                return
            # If it's different, log a warning but still update
            logger.warning(f"Overwriting existing field mapping: {schema_name}.{param_name} -> {existing_field} with {field_name}")

        # Register the mapping
        cls._field_mappings[schema_name][param_name] = field_name
        logger.info(f"Registered field mapping: {schema_name}.{param_name} -> {field_name}")

    @classmethod
    def get_schema(cls, schema_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a database schema by name.

        Args:
            schema_name: Name of the schema

        Returns:
            The schema if found, None otherwise
        """
        return cls._db_schemas.get(schema_name)

    @classmethod
    def get_field_mapping(cls, schema_name: str, param_name: str) -> Optional[str]:
        """
        Get the database field name for a tool parameter.

        Args:
            schema_name: Name of the schema
            param_name: Name of the tool parameter

        Returns:
            The field name if found, None otherwise
        """
        if schema_name not in cls._field_mappings:
            return None

        return cls._field_mappings[schema_name].get(param_name)

    @classmethod
    def resolve_field(cls, schema_name: str, param_name: str, default: Optional[str] = None) -> str:
        """
        Resolve a parameter name to a database field name.

        Args:
            schema_name: Name of the schema
            param_name: Name of the tool parameter
            default: Default field name to use if mapping not found

        Returns:
            The resolved field name
        """
        field_name = cls.get_field_mapping(schema_name, param_name)
        if field_name is not None:
            return field_name

        # If no mapping is found, use the parameter name as the field name
        return default or param_name

    @classmethod
    def validate_schema(cls, schema_name: str) -> Tuple[bool, List[str]]:
        """
        Validate a database schema.

        Args:
            schema_name: Name of the schema

        Returns:
            Tuple of (is_valid, error_messages)
        """
        schema = cls.get_schema(schema_name)
        if not schema:
            return False, [f"Schema '{schema_name}' not found"]

        errors = []

        # Check for required fields
        if "fields" not in schema:
            errors.append(f"Schema '{schema_name}' is missing 'fields' definition")
            return False, errors

        # Check field definitions
        fields = schema.get("fields", {})
        for field_name, field_def in fields.items():
            if not isinstance(field_def, dict):
                errors.append(f"Field '{field_name}' in schema '{schema_name}' has invalid definition")
                continue

            if "type" not in field_def:
                errors.append(f"Field '{field_name}' in schema '{schema_name}' is missing 'type' definition")

        # Check field mappings
        if schema_name in cls._field_mappings:
            for param_name, field_name in cls._field_mappings[schema_name].items():
                if field_name not in fields:
                    errors.append(f"Field mapping '{param_name}' -> '{field_name}' references non-existent field in schema '{schema_name}'")

        return len(errors) == 0, errors

    @classmethod
    def validate_all_schemas(cls) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate all registered schemas.

        Returns:
            Tuple of (all_valid, {schema_name: error_messages})
        """
        all_valid = True
        all_errors = {}

        for schema_name in cls._db_schemas:
            is_valid, errors = cls.validate_schema(schema_name)
            if not is_valid:
                all_valid = False
                all_errors[schema_name] = errors

        return all_valid, all_errors

    @classmethod
    def reload_all_schemas(cls) -> bool:
        """
        Reload all schemas from their source files.

        Returns:
            True if all schemas were reloaded successfully, False otherwise
        """
        success = True
        for schema_name, schema_file in cls._schema_files.items():
            if not cls.load_schema(schema_name, schema_file):
                success = False

        return success

    @classmethod
    def get_field_info(cls, schema_name: str, field_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific field in a schema.

        Args:
            schema_name: Name of the schema
            field_name: Name of the field

        Returns:
            Field information if found, None otherwise
        """
        schema = cls.get_schema(schema_name)
        if not schema:
            return None

        fields = schema.get("fields", {})
        return fields.get(field_name)

    @classmethod
    def list_schemas(cls) -> List[str]:
        """
        List all registered schemas.

        Returns:
            List of schema names
        """
        return list(cls._db_schemas.keys())

    @classmethod
    def list_fields(cls, schema_name: str) -> List[str]:
        """
        List all fields in a schema.

        Args:
            schema_name: Name of the schema

        Returns:
            List of field names
        """
        schema = cls.get_schema(schema_name)
        if not schema:
            return []

        return list(schema.get("fields", {}).keys())

    @classmethod
    def get_field_aliases(cls, schema_name: str, field_name: str) -> List[str]:
        """
        Get all aliases for a field.

        Args:
            schema_name: Name of the schema
            field_name: Name of the field

        Returns:
            List of field aliases
        """
        field_info = cls.get_field_info(schema_name, field_name)
        if not field_info:
            return []

        return field_info.get("aliases", [])

    @classmethod
    def find_field_by_alias(cls, schema_name: str, alias: str) -> Optional[str]:
        """
        Find a field by its alias.

        Args:
            schema_name: Name of the schema
            alias: Alias to search for

        Returns:
            Field name if found, None otherwise
        """
        schema = cls.get_schema(schema_name)
        if not schema:
            return None

        fields = schema.get("fields", {})
        for field_name, field_info in fields.items():
            aliases = field_info.get("aliases", [])
            if alias in aliases or alias == field_name:
                return field_name

        return None
