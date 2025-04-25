"""
Test Unified Configuration

This script tests the unified configuration system with all the new configuration types.
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    # Import the ConfigProvider and configuration classes
    from src.sec_filing_analyzer.config import (
        AgentConfig,
        ConfigProvider,
        ETLConfig,
        Neo4jConfig,
        StorageConfig,
        StreamlitConfig,
        VectorStoreConfig,
    )

    # Initialize the ConfigProvider
    ConfigProvider.initialize()

    # Test getting configuration instances
    print("Testing configuration instances:")
    print(f"Neo4jConfig: {ConfigProvider.get_config(Neo4jConfig)}")
    print(f"StorageConfig: {ConfigProvider.get_config(StorageConfig)}")
    print(f"ETLConfig: {ConfigProvider.get_config(ETLConfig)}")
    print(f"AgentConfig: {ConfigProvider.get_config(AgentConfig)}")
    print(f"VectorStoreConfig: {ConfigProvider.get_config(VectorStoreConfig)}")
    print(f"StreamlitConfig: {ConfigProvider.get_config(StreamlitConfig)}")
    print()

    # Test getting agent configurations
    print("Testing agent configurations:")
    agent_types = ConfigProvider.get_all_agent_types()
    print(f"Available agent types: {agent_types}")
    for agent_type in agent_types:
        config = ConfigProvider.get_agent_config(agent_type)
        print(f"Configuration for {agent_type}:")
        print(f"  Model: {config.get('model', 'Not specified')}")
        print(f"  Temperature: {config.get('temperature', 'Not specified')}")
        print(f"  Max Tokens: {config.get('max_tokens', 'Not specified')}")
    print()

    # Test getting tool schemas
    print("Testing tool schemas:")
    tool_schemas = ConfigProvider.get_all_tool_schemas()
    print(f"Available tool schemas: {list(tool_schemas.keys())}")
    for tool_name, schema in tool_schemas.items():
        print(f"Schema for {tool_name}:")
        print(f"  Parameters: {list(schema.keys())}")
    print()

    # Test validation
    print("Testing configuration validation:")
    config = ConfigProvider.get_agent_config("coordinator")
    print(f"Coordinator config validation: {ConfigProvider.validate_config(config, 'agent')}")

    etl_config = vars(ConfigProvider.get_config(ETLConfig))
    print(f"ETL config validation: {ConfigProvider.validate_config(etl_config, 'etl')}")

    vector_store_config = vars(ConfigProvider.get_config(VectorStoreConfig))
    print(f"Vector store config validation: {ConfigProvider.validate_config(vector_store_config, 'vector_store')}")

    streamlit_config = vars(ConfigProvider.get_config(StreamlitConfig))
    print(f"Streamlit config validation: {ConfigProvider.validate_config(streamlit_config, 'streamlit')}")
    print()

    print("All tests completed successfully!")

except Exception as e:
    print(f"Error testing unified configuration: {str(e)}")
    import traceback

    traceback.print_exc()
