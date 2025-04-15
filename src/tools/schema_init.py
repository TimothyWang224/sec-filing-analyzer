"""
Schema Initialization

This module initializes the schema registry with database schemas.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from .schema_registry import SchemaRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_schemas(schema_dir: Optional[str] = None) -> bool:
    """
    Initialize the schema registry with database schemas.

    Args:
        schema_dir: Optional directory containing schema files

    Returns:
        True if initialization was successful, False otherwise
    """
    # Default schema directory
    if schema_dir is None:
        # Try to find the schema directory
        possible_dirs = [
            "data/schemas",
            "../data/schemas",
            "../../data/schemas",
            os.path.join(os.path.dirname(__file__), "../../data/schemas")
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                schema_dir = dir_path
                break
        
        if schema_dir is None:
            logger.error("Schema directory not found")
            return False
    
    # Ensure schema directory exists
    if not os.path.exists(schema_dir) or not os.path.isdir(schema_dir):
        logger.error(f"Schema directory not found: {schema_dir}")
        return False
    
    # Load all schema files
    schema_files = list(Path(schema_dir).glob("*.json"))
    if not schema_files:
        logger.warning(f"No schema files found in {schema_dir}")
        return False
    
    # Load each schema file
    success = True
    for schema_file in schema_files:
        schema_name = schema_file.stem
        if not SchemaRegistry.load_schema(schema_name, str(schema_file)):
            success = False
    
    # Validate all schemas
    is_valid, errors = SchemaRegistry.validate_all_schemas()
    if not is_valid:
        logger.error(f"Schema validation errors: {errors}")
        success = False
    
    return success

def get_loaded_schemas() -> List[str]:
    """
    Get a list of loaded schemas.

    Returns:
        List of schema names
    """
    return SchemaRegistry.list_schemas()

def get_schema_fields(schema_name: str) -> List[str]:
    """
    Get a list of fields in a schema.

    Args:
        schema_name: Name of the schema

    Returns:
        List of field names
    """
    return SchemaRegistry.list_fields(schema_name)

# Initialize schemas when module is imported
initialize_schemas()
