# Schema-Driven Tools

This document describes the schema-driven approach to tool implementation in the SEC Filing Analyzer.

**Status: All tools have been updated to use the schema-driven approach.**

## Overview

The schema-driven approach provides a centralized way to define database schemas and their mappings to tool parameters. This ensures consistency between tool parameters and database fields, and makes it easier to handle changes to database schemas.

## Components

### SchemaRegistry

The `SchemaRegistry` is responsible for managing database schemas and field mappings. It provides methods for:

- Loading schemas from JSON files
- Registering field mappings
- Validating schemas and mappings
- Resolving field names from parameter names

### ToolRegistry

The `ToolRegistry` has been enhanced to integrate with the `SchemaRegistry`. It now:

- Stores schema mappings for tools
- Validates schema mappings
- Provides methods to access schema mappings

### Tool Base Class

The `Tool` base class has been updated to support dynamic parameter resolution. It now:

- Resolves parameters to database field names
- Delegates execution to the `_execute` method
- Provides a consistent interface for tool implementation

### Tool Decorator

A new `@tool` decorator has been added to simplify tool registration and configuration. It:

- Registers the tool with the `ToolRegistry`
- Sets up schema mappings
- Configures tool metadata

## Usage

### Defining Database Schemas

Database schemas are defined in JSON files in the `data/schemas` directory. Each schema file defines the fields, indexes, and relationships for a database table.

The following schemas have been defined:

1. `financial_facts.json` - Schema for financial facts table
2. `companies.json` - Schema for companies table
3. `filings.json` - Schema for SEC filings table
4. `document_chunks.json` - Schema for document chunks used in semantic search
5. `graph_entities.json` - Schema for entities in the graph database

Example schema file (`financial_facts.json`):

```json
{
  "name": "financial_facts",
  "description": "Schema for financial facts table in DuckDB",
  "version": "1.0",
  "fields": {
    "ticker": {
      "type": "string",
      "description": "Company ticker symbol",
      "aliases": ["company_ticker", "symbol"],
      "required": true
    },
    "metric_name": {
      "type": "string",
      "description": "Name of the financial metric",
      "aliases": ["metric", "financial_metric"],
      "required": true
    },
    // ... more fields ...
  },
  // ... indexes and relationships ...
}
```

### Implementing a Tool

To implement a tool using the schema-driven approach:

1. Use the `@tool` decorator to register the tool and configure schema mappings
2. Implement the `_execute` method instead of `execute`
3. Use the schema mappings to access database fields

Example:

```python
@tool(
    name="sec_financial_data",
    tags=["sec", "financial", "data"],
    compact_description="Query financial metrics and facts from SEC filings",
    db_schema="financial_facts",
    parameter_mappings={
        "ticker": "ticker",
        "metric": "metric_name",
        "start_date": "period_start_date",
        "end_date": "period_end_date",
        "filing_type": "filing_type"
    }
)
class SECFinancialDataTool(Tool):
    """Tool for querying financial data from SEC filings."""

    async def _execute(self, query_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a financial data query."""
        # Implementation...
```

### Parameter Resolution

When a tool is executed, the parameters are automatically resolved to database field names based on the schema mappings. For example:

```python
# Original parameters
params = {
    "query_type": "financial_facts",
    "parameters": {
        "ticker": "AAPL",
        "metric": "Revenue",
        "start_date": "2022-01-01",
        "end_date": "2023-01-01"
    }
}

# Resolved parameters
resolved_params = {
    "query_type": "financial_facts",
    "parameters": {
        "ticker": "AAPL",
        "metric_name": "Revenue",
        "period_start_date": "2022-01-01",
        "period_end_date": "2023-01-01"
    }
}
```

## Benefits

The schema-driven approach provides several benefits:

1. **Single Source of Truth**: Database schemas are the definitive source of truth for field definitions
2. **Maintainability**: Changes to database fields only require updating schema files
3. **Consistency**: Centralized validation ensures consistency between tools and database
4. **Extensibility**: Easy to add new tools and schemas
5. **Robustness**: System can handle changes to database schemas gracefully

## Testing

Two test scripts are provided to demonstrate the schema-driven approach:

```bash
# Test a single tool (SECFinancialDataTool)
python scripts/test_schema_driven_tools.py

# Test all tools
python scripts/test_all_schema_driven_tools.py
```

These scripts test the `SchemaRegistry`, `ToolRegistry`, and all tool implementations.

## Implemented Tools

The following tools have been updated to use the schema-driven approach:

1. **SECDataTool** - For retrieving raw SEC filing data
   - Not using schema mappings due to custom parameter structure

2. **SECSemanticSearchTool** - For semantic search on SEC filings
   - Not using schema mappings due to complex parameter structure

3. **SECGraphQueryTool** - For querying the SEC filing graph database
   - Not using schema mappings due to nested parameter structure

4. **SECFinancialDataTool** - For querying financial data from SEC filings
   - Schema: `financial_facts`
   - Parameter mappings: ticker, metric -> metric_name, start_date -> period_start_date, end_date -> period_end_date, filing_type

5. **ToolDetailsTool** - For getting detailed information about other tools
   - No schema (metadata tool)
