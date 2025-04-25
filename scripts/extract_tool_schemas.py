"""
Extract Tool Schemas

This script extracts tool parameter schemas from the tool_parameter_helper.py file
and saves them as JSON files in the data/schemas directory.
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    # Import the tool parameter schemas
    from src.tools.tool_parameter_helper import TOOL_PARAMETER_SCHEMAS

    # Create the schemas directory if it doesn't exist
    schemas_dir = Path("data/schemas")
    schemas_dir.mkdir(parents=True, exist_ok=True)

    # Save each schema as a JSON file
    for tool_name, schema in TOOL_PARAMETER_SCHEMAS.items():
        schema_file = schemas_dir / f"{tool_name}.json"

        # Save the schema to a JSON file
        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=2)

        print(f"Saved schema for tool: {tool_name} to {schema_file}")

    print(f"Successfully extracted {len(TOOL_PARAMETER_SCHEMAS)} tool schemas to {schemas_dir}")

except Exception as e:
    print(f"Error extracting tool schemas: {str(e)}")
    import traceback

    traceback.print_exc()
