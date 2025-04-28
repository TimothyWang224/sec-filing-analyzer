"""
Unit tests for the SchemaRegistry class.
"""

import json
from unittest.mock import mock_open, patch


from src.tools.schema_registry import SchemaRegistry


class TestSchemaRegistry:
    """Test suite for the SchemaRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear the registry before each test
        SchemaRegistry._db_schemas = {}
        SchemaRegistry._field_mappings = {}
        SchemaRegistry._schema_files = {}

    def test_load_schema(self):
        """Test loading a schema from a JSON file."""
        # Mock the open function to return a schema
        schema_json = json.dumps(
            {
                "name": "test_schema",
                "fields": {
                    "field1": {"type": "string", "description": "Field 1"},
                    "field2": {"type": "number", "description": "Field 2"},
                },
            }
        )

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=schema_json)),
        ):
            # Load the schema
            result = SchemaRegistry.load_schema("test_schema", "test_schema.json")

            # Check that the schema was loaded
            assert result
            assert "test_schema" in SchemaRegistry._db_schemas
            assert "test_schema" in SchemaRegistry._schema_files
            assert (
                SchemaRegistry._db_schemas["test_schema"]["fields"]["field1"]["type"]
                == "string"
            )

    def test_load_schema_file_not_found(self):
        """Test loading a schema from a file that doesn't exist."""
        # Mock the os.path.exists function to return False
        with patch("os.path.exists", return_value=False):
            # Load the schema
            result = SchemaRegistry.load_schema("test_schema", "nonexistent.json")

            # Check that the schema was not loaded
            assert not result
            assert "test_schema" not in SchemaRegistry._db_schemas

    def test_register_field_mapping(self):
        """Test registering a field mapping."""
        # Register a field mapping
        SchemaRegistry.register_field_mapping("test_schema", "param1", "field1")

        # Check that the mapping was registered
        assert "test_schema" in SchemaRegistry._field_mappings
        assert SchemaRegistry._field_mappings["test_schema"]["param1"] == "field1"

    def test_register_field_mapping_overwrite(self):
        """Test overwriting an existing field mapping."""
        # Register a field mapping
        SchemaRegistry.register_field_mapping("test_schema", "param1", "field1")

        # Register a different mapping for the same parameter
        with patch("logging.Logger.warning") as mock_warning:
            SchemaRegistry.register_field_mapping("test_schema", "param1", "field2")

            # Check that a warning was logged
            mock_warning.assert_called_once()

        # Check that the mapping was overwritten
        assert SchemaRegistry._field_mappings["test_schema"]["param1"] == "field2"

    def test_get_schema(self):
        """Test getting a schema."""
        # Add a schema to the registry
        SchemaRegistry._db_schemas["test_schema"] = {
            "name": "test_schema",
            "fields": {
                "field1": {"type": "string", "description": "Field 1"},
                "field2": {"type": "number", "description": "Field 2"},
            },
        }

        # Get the schema
        schema = SchemaRegistry.get_schema("test_schema")

        # Check that the schema was retrieved
        assert schema is not None
        assert schema["fields"]["field1"]["type"] == "string"

    def test_get_nonexistent_schema(self):
        """Test getting a schema that doesn't exist."""
        # Get a nonexistent schema
        schema = SchemaRegistry.get_schema("nonexistent_schema")

        # Check that None was returned
        assert schema is None

    def test_get_field_mapping(self):
        """Test getting a field mapping."""
        # Add a field mapping to the registry
        SchemaRegistry._field_mappings["test_schema"] = {
            "param1": "field1",
            "param2": "field2",
        }

        # Get the field mapping
        field_name = SchemaRegistry.get_field_mapping("test_schema", "param1")

        # Check that the field name was retrieved
        assert field_name == "field1"

    def test_get_nonexistent_field_mapping(self):
        """Test getting a field mapping that doesn't exist."""
        # Get a nonexistent field mapping
        field_name = SchemaRegistry.get_field_mapping("nonexistent_schema", "param1")

        # Check that None was returned
        assert field_name is None

    def test_resolve_field_with_mapping(self):
        """Test resolving a field name with a mapping."""
        # Add a field mapping to the registry
        SchemaRegistry._field_mappings["test_schema"] = {
            "param1": "field1",
            "param2": "field2",
        }

        # Resolve the field name
        field_name = SchemaRegistry.resolve_field("test_schema", "param1")

        # Check that the field name was resolved
        assert field_name == "field1"

    def test_resolve_field_without_mapping(self):
        """Test resolving a field name without a mapping."""
        # Resolve the field name
        field_name = SchemaRegistry.resolve_field("test_schema", "param1")

        # Check that the parameter name was returned
        assert field_name == "param1"

    def test_resolve_field_with_default(self):
        """Test resolving a field name with a default value."""
        # Resolve the field name with a default
        field_name = SchemaRegistry.resolve_field(
            "test_schema", "param1", default="default_field"
        )

        # Check that the default was returned
        assert field_name == "default_field"

    def test_validate_schema_valid(self):
        """Test validating a valid schema."""
        # Add a schema to the registry
        SchemaRegistry._db_schemas["test_schema"] = {
            "name": "test_schema",
            "fields": {
                "field1": {"type": "string", "description": "Field 1"},
                "field2": {"type": "number", "description": "Field 2"},
            },
        }

        # Add field mappings
        SchemaRegistry._field_mappings["test_schema"] = {
            "param1": "field1",
            "param2": "field2",
        }

        # Validate the schema
        is_valid, errors = SchemaRegistry.validate_schema("test_schema")

        # Check that the schema is valid
        assert is_valid
        assert not errors

    def test_validate_schema_missing_fields(self):
        """Test validating a schema with missing fields."""
        # Add a schema to the registry without fields
        SchemaRegistry._db_schemas["test_schema"] = {"name": "test_schema"}

        # Validate the schema
        is_valid, errors = SchemaRegistry.validate_schema("test_schema")

        # Check that the schema is invalid
        assert not is_valid
        assert len(errors) == 1
        assert "missing 'fields'" in errors[0]

    def test_validate_schema_invalid_field_definition(self):
        """Test validating a schema with an invalid field definition."""
        # Add a schema to the registry with an invalid field definition
        SchemaRegistry._db_schemas["test_schema"] = {
            "name": "test_schema",
            "fields": {
                "field1": "invalid"  # Not a dictionary
            },
        }

        # Validate the schema
        is_valid, errors = SchemaRegistry.validate_schema("test_schema")

        # Check that the schema is invalid
        assert not is_valid
        assert len(errors) == 1
        assert "invalid definition" in errors[0]

    def test_validate_schema_missing_type(self):
        """Test validating a schema with a field missing a type."""
        # Add a schema to the registry with a field missing a type
        SchemaRegistry._db_schemas["test_schema"] = {
            "name": "test_schema",
            "fields": {
                "field1": {"description": "Field 1"}  # Missing type
            },
        }

        # Validate the schema
        is_valid, errors = SchemaRegistry.validate_schema("test_schema")

        # Check that the schema is invalid
        assert not is_valid
        assert len(errors) == 1
        assert "missing 'type'" in errors[0]

    def test_validate_schema_invalid_mapping(self):
        """Test validating a schema with an invalid field mapping."""
        # Add a schema to the registry
        SchemaRegistry._db_schemas["test_schema"] = {
            "name": "test_schema",
            "fields": {"field1": {"type": "string", "description": "Field 1"}},
        }

        # Add an invalid field mapping
        SchemaRegistry._field_mappings["test_schema"] = {"param1": "nonexistent_field"}

        # Validate the schema
        is_valid, errors = SchemaRegistry.validate_schema("test_schema")

        # Check that the schema is invalid
        assert not is_valid
        assert len(errors) == 1
        assert "nonexistent_field" in errors[0]

    def test_validate_all_schemas(self):
        """Test validating all schemas."""
        # Add two schemas to the registry
        SchemaRegistry._db_schemas["schema1"] = {
            "name": "schema1",
            "fields": {"field1": {"type": "string", "description": "Field 1"}},
        }
        SchemaRegistry._db_schemas["schema2"] = {
            "name": "schema2",
            "fields": {"field2": {"type": "number", "description": "Field 2"}},
        }

        # Validate all schemas
        is_valid, all_errors = SchemaRegistry.validate_all_schemas()

        # Check that all schemas are valid
        assert is_valid
        assert not all_errors

    def test_reload_all_schemas(self):
        """Test reloading all schemas."""
        # Add schema files to the registry
        SchemaRegistry._schema_files["schema1"] = "schema1.json"
        SchemaRegistry._schema_files["schema2"] = "schema2.json"

        # Mock the load_schema method
        with patch(
            "src.tools.schema_registry.SchemaRegistry.load_schema", return_value=True
        ):
            # Reload all schemas
            result = SchemaRegistry.reload_all_schemas()

            # Check that all schemas were reloaded
            assert result

    def test_get_field_info(self):
        """Test getting field information."""
        # Add a schema to the registry
        SchemaRegistry._db_schemas["test_schema"] = {
            "name": "test_schema",
            "fields": {"field1": {"type": "string", "description": "Field 1"}},
        }

        # Get field information
        field_info = SchemaRegistry.get_field_info("test_schema", "field1")

        # Check that the field information was retrieved
        assert field_info is not None
        assert field_info["type"] == "string"
        assert field_info["description"] == "Field 1"

    def test_list_schemas(self):
        """Test listing all schemas."""
        # Add two schemas to the registry
        SchemaRegistry._db_schemas["schema1"] = {}
        SchemaRegistry._db_schemas["schema2"] = {}

        # List all schemas
        schemas = SchemaRegistry.list_schemas()

        # Check that both schemas are in the list
        assert "schema1" in schemas
        assert "schema2" in schemas

    def test_list_fields(self):
        """Test listing all fields in a schema."""
        # Add a schema to the registry
        SchemaRegistry._db_schemas["test_schema"] = {
            "name": "test_schema",
            "fields": {"field1": {"type": "string"}, "field2": {"type": "number"}},
        }

        # List all fields
        fields = SchemaRegistry.list_fields("test_schema")

        # Check that both fields are in the list
        assert "field1" in fields
        assert "field2" in fields

    def test_get_field_aliases(self):
        """Test getting field aliases."""
        # Add a schema to the registry with field aliases
        SchemaRegistry._db_schemas["test_schema"] = {
            "name": "test_schema",
            "fields": {"field1": {"type": "string", "aliases": ["alias1", "alias2"]}},
        }

        # Get field aliases
        aliases = SchemaRegistry.get_field_aliases("test_schema", "field1")

        # Check that the aliases were retrieved
        assert "alias1" in aliases
        assert "alias2" in aliases

    def test_find_field_by_alias(self):
        """Test finding a field by its alias."""
        # Add a schema to the registry with field aliases
        SchemaRegistry._db_schemas["test_schema"] = {
            "name": "test_schema",
            "fields": {"field1": {"type": "string", "aliases": ["alias1", "alias2"]}},
        }

        # Find field by alias
        field_name = SchemaRegistry.find_field_by_alias("test_schema", "alias1")

        # Check that the field was found
        assert field_name == "field1"

    def test_find_field_by_name(self):
        """Test finding a field by its name."""
        # Add a schema to the registry
        SchemaRegistry._db_schemas["test_schema"] = {
            "name": "test_schema",
            "fields": {"field1": {"type": "string"}},
        }

        # Find field by name
        field_name = SchemaRegistry.find_field_by_alias("test_schema", "field1")

        # Check that the field was found
        assert field_name == "field1"
