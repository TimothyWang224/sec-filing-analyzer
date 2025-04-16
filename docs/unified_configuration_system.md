# Unified Configuration System

This document describes the unified configuration system for the SEC Filing Analyzer project.

## Overview

The SEC Filing Analyzer uses a unified configuration system that provides a centralized way to manage configuration values across the application. The configuration system is designed to be flexible, allowing configuration values to be specified in multiple ways:

1. **Configuration Files**: Configuration values can be specified in JSON files.
2. **Environment Variables**: Configuration values can be specified using environment variables.
3. **Direct Parameters**: Configuration values can be passed directly to functions and classes.

The configuration system uses a hierarchical approach, with more specific configurations overriding more general ones.

## Configuration Provider

The `ConfigProvider` class is the central component of the configuration system. It provides a unified interface for accessing configuration values from various sources.

### Usage

```python
from sec_filing_analyzer.config import ConfigProvider, ETLConfig

# Initialize the ConfigProvider
ConfigProvider.initialize()

# Get a configuration instance by type
etl_config = ConfigProvider.get_config(ETLConfig)

# Get configuration for a specific agent type
agent_config = ConfigProvider.get_agent_config("financial_analyst")

# Get a tool schema
tool_schema = ConfigProvider.get_tool_schema("sec_semantic_search")
```

### Configuration Types

The configuration system supports the following configuration types:

#### Neo4jConfig

Configuration for Neo4j database connection.

```python
from sec_filing_analyzer.config import ConfigProvider, Neo4jConfig

neo4j_config = ConfigProvider.get_config(Neo4jConfig)
print(f"Neo4j URL: {neo4j_config.url}")
```

#### StorageConfig

Configuration for storage settings.

```python
from sec_filing_analyzer.config import ConfigProvider, StorageConfig

storage_config = ConfigProvider.get_config(StorageConfig)
print(f"Vector store path: {storage_config.vector_store_path}")
```

#### ETLConfig

Configuration for the ETL pipeline.

```python
from sec_filing_analyzer.config import ConfigProvider, ETLConfig

etl_config = ConfigProvider.get_config(ETLConfig)
print(f"Chunk size: {etl_config.chunk_size}")
```

#### AgentConfig

Configuration for agent parameters.

```python
from sec_filing_analyzer.config import ConfigProvider, AgentConfig

agent_config = ConfigProvider.get_config(AgentConfig)
print(f"Max iterations: {agent_config.max_iterations}")
```

#### VectorStoreConfig

Configuration for vector store.

```python
from sec_filing_analyzer.config import ConfigProvider, VectorStoreConfig

vector_store_config = ConfigProvider.get_config(VectorStoreConfig)
print(f"Index type: {vector_store_config.index_type}")
```

#### StreamlitConfig

Configuration for Streamlit applications.

```python
from sec_filing_analyzer.config import ConfigProvider, StreamlitConfig

streamlit_config = ConfigProvider.get_config(StreamlitConfig)
print(f"Port: {streamlit_config.port}")
```

### Agent Configurations

The configuration system provides agent-specific configurations for different agent types.

```python
from sec_filing_analyzer.config import ConfigProvider

# Get all available agent types
agent_types = ConfigProvider.get_all_agent_types()
print(f"Available agent types: {agent_types}")

# Get configuration for a specific agent type
config = ConfigProvider.get_agent_config("financial_analyst")
print(f"Model: {config.get('model')}")
```

### Tool Schemas

The configuration system provides schemas for tool parameters.

```python
from sec_filing_analyzer.config import ConfigProvider

# Get all available tool schemas
tool_schemas = ConfigProvider.get_all_tool_schemas()
print(f"Available tool schemas: {list(tool_schemas.keys())}")

# Get schema for a specific tool
schema = ConfigProvider.get_tool_schema("sec_semantic_search")
print(f"Parameters: {list(schema.keys())}")
```

## Configuration Files

The configuration system uses JSON files to store configuration values. The main configuration file is `data/config/etl_config.json`.

### Example Configuration File

```json
{
  "vector_store": {
    "type": "optimized",
    "path": "data/vector_store",
    "index_type": "hnsw",
    "use_gpu": false,
    "hnsw_m": 32,
    "hnsw_ef_construction": 400,
    "hnsw_ef_search": 200
  },
  "graph_store": {
    "type": "neo4j",
    "url": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "password",
    "database": "neo4j"
  },
  "duckdb": {
    "path": "data/financial_data.duckdb"
  },
  "etl_pipeline": {
    "process_semantic": true,
    "process_quantitative": true,
    "use_parallel": true,
    "max_workers": 4,
    "batch_size": 50,
    "rate_limit": 0.2,
    "max_retries": 3,
    "delay_between_companies": 1,
    "chunk_size": 1024,
    "chunk_overlap": 50,
    "embedding_model": "text-embedding-3-small"
  },
  "agent": {
    "max_iterations": 3,
    "max_planning_iterations": 2,
    "max_execution_iterations": 3,
    "max_refinement_iterations": 1,
    "max_tool_retries": 2,
    "tools_per_iteration": 1,
    "circuit_breaker_threshold": 3,
    "circuit_breaker_reset_timeout": 300,
    "max_duration_seconds": 180,
    "enable_dynamic_termination": false,
    "min_confidence_threshold": 0.8,
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0.7,
    "llm_max_tokens": 4000
  },
  "streamlit": {
    "server": {
      "port": 8501,
      "headless": true,
      "enable_cors": true,
      "enable_xsrf_protection": false,
      "max_upload_size": 200,
      "base_url_path": ""
    },
    "theme": {
      "base": "light"
    },
    "ui": {
      "hide_top_bar": false
    },
    "client": {
      "show_error_details": true,
      "toolbar_mode": "auto",
      "caching": false
    }
  }
}
```

### Tool Schema Files

Tool schemas are stored in JSON files in the `data/schemas` directory. Each tool has its own schema file.

Example tool schema file (`data/schemas/sec_semantic_search.json`):

```json
{
  "query": {
    "type": "string",
    "required": true,
    "description": "The search query to execute"
  },
  "companies": {
    "type": "array",
    "required": false,
    "description": "List of company ticker symbols to search"
  },
  "top_k": {
    "type": "integer",
    "required": false,
    "default": 5,
    "description": "Number of results to return"
  },
  "filing_types": {
    "type": "array",
    "required": false,
    "description": "List of filing types to search"
  },
  "date_range": {
    "type": "array",
    "required": false,
    "description": "Date range to search [start_date, end_date]"
  },
  "sections": {
    "type": "array",
    "required": false,
    "description": "List of document sections to search"
  }
}
```

## Environment Variables

The configuration system supports environment variables for all configuration values. Environment variables are loaded from a `.env` file if it exists.

### Example Environment Variables

```
# Neo4j Configuration
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_URL=bolt://localhost:7687
NEO4J_DATABASE=neo4j

# Storage Configuration
GRAPH_MAX_CLUSTER_SIZE=5
USE_NEO4J=true
VECTOR_STORE_TYPE=optimized

# ETL Configuration
SEC_FILINGS_DIR=data/filings
SEC_FILING_TYPES=10-K,10-Q,8-K
SEC_MAX_RETRIES=3
SEC_TIMEOUT=30
CHUNK_SIZE=1024
CHUNK_OVERLAP=50
EMBEDDING_MODEL=text-embedding-3-small
USE_PARALLEL=true
MAX_WORKERS=4
BATCH_SIZE=100
RATE_LIMIT=0.1

# Agent Configuration
AGENT_MAX_ITERATIONS=3
AGENT_MAX_PLANNING_ITERATIONS=2
AGENT_MAX_EXECUTION_ITERATIONS=3
AGENT_MAX_REFINEMENT_ITERATIONS=1
AGENT_MAX_TOOL_RETRIES=2
AGENT_TOOLS_PER_ITERATION=1
AGENT_MAX_DURATION_SECONDS=180
AGENT_ENABLE_DYNAMIC_TERMINATION=false
AGENT_MIN_CONFIDENCE_THRESHOLD=0.8
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_LLM_TEMPERATURE=0.7
DEFAULT_LLM_MAX_TOKENS=4000

# Vector Store Configuration
VECTOR_STORE_TYPE=optimized
VECTOR_STORE_PATH=data/vector_store
VECTOR_INDEX_TYPE=hnsw
VECTOR_USE_GPU=false
VECTOR_HNSW_M=32
VECTOR_HNSW_EF_CONSTRUCTION=400
VECTOR_HNSW_EF_SEARCH=200

# Streamlit Configuration
STREAMLIT_PORT=8501
STREAMLIT_HEADLESS=true
STREAMLIT_ENABLE_CORS=true
STREAMLIT_THEME_BASE=light
```

## Configuration Hierarchy

The configuration values are resolved in the following order:

1. Values passed directly to functions and classes
2. Values from agent-specific configurations in `llm_config.py`
3. Values from the global `AGENT_CONFIG` in `config.py`
4. Values from external configuration files
5. Values from environment variables
6. Default hardcoded values

This hierarchy allows for flexible configuration at different levels.

## Validation

The configuration system includes validation logic to ensure that configuration values are valid.

```python
from sec_filing_analyzer.config import ConfigProvider

# Validate a configuration
config = ConfigProvider.get_agent_config("coordinator")
is_valid = ConfigProvider.validate_config(config, 'agent')
print(f"Is valid: {is_valid}")
```

## Utility Scripts

The configuration system includes several utility scripts for managing configuration values:

### extract_tool_schemas.py

Extracts tool parameter schemas from the `tool_parameter_helper.py` file and saves them as JSON files in the `data/schemas` directory.

```bash
python scripts/extract_tool_schemas.py
```

### extract_streamlit_config.py

Extracts the Streamlit configuration from `.streamlit/config.toml` and updates the unified configuration file.

```bash
python scripts/extract_streamlit_config.py
```

### extract_vector_store_params.py

Extracts vector store index parameters from the `params.json` files and updates the unified configuration file.

```bash
python scripts/extract_vector_store_params.py
```

### update_unified_config.py

Updates the unified configuration file with all the settings from various sources.

```bash
python scripts/update_unified_config.py
```

### test_unified_config.py

Tests the unified configuration system with all the new configuration types.

```bash
python scripts/test_unified_config.py
```

## Integration with Tools

The configuration system is integrated with the tool system to provide tool schemas and parameter mappings.

```python
from sec_filing_analyzer.config import ConfigProvider
from src.tools.schema_registry import SchemaRegistry

# Get schema registry from ConfigProvider
schema_registry = ConfigProvider.get_config(SchemaRegistry)

# Get mappings for a schema
mappings = schema_registry.get_field_mappings("financial_facts")
print(f"Mappings: {mappings}")
```

## Future Improvements

1. **Configuration UI**: Create a simple UI for editing configuration files to make it easier for users to customize the system.
2. **Configuration Versioning**: Add version information to configuration files to track changes over time.
3. **Configuration Validation**: Enhance validation logic to provide more detailed error messages.
4. **Configuration Documentation**: Generate documentation for configuration values from the code.
5. **Configuration Inheritance**: Add support for configuration inheritance to reduce duplication.
6. **Configuration Encryption**: Add support for encrypting sensitive configuration values.
7. **Configuration Backup**: Add support for backing up configuration files before making changes.
8. **Configuration Diff**: Add support for showing differences between configuration files.
9. **Configuration Import/Export**: Add support for importing and exporting configuration files.
10. **Configuration Templates**: Add support for configuration templates for different use cases.
