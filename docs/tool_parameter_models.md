# Tool Parameter Models

This document describes the parameter models used by tools in the SEC Filing Analyzer system.

## Overview

The SEC Filing Analyzer uses Pydantic models to define and validate tool parameters. Each tool defines a set of parameter models for different query types, which are used to validate parameters before tool execution.

## Base Models

### ToolInput

The `ToolInput` model is the base model for all tool inputs:

```python
class ToolInput(BaseModel):
    """Base model for tool inputs."""
    query_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
```

- `query_type`: The type of query to execute
- `parameters`: The parameters for the query

## Financial Data Models

### FinancialFactsParams

The `FinancialFactsParams` model defines parameters for financial facts queries:

```python
class FinancialFactsParams(BaseModel):
    """Parameters for financial facts queries."""
    ticker: str
    metrics: List[str]
    start_date: str
    end_date: str
    filing_type: Optional[str] = None
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v):
        # Simple validation for YYYY-MM-DD format
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Date must be a non-empty string")
        
        # Check if it's already a date object
        if isinstance(v, date):
            return v.isoformat()
            
        # Basic format check
        parts = v.split('-')
        if len(parts) != 3:
            raise ValueError("Date must be in YYYY-MM-DD format")
            
        return v
```

### MetricsParams

The `MetricsParams` model defines parameters for metrics queries:

```python
class MetricsParams(BaseModel):
    """Parameters for metrics queries."""
    ticker: str
    year: Optional[int] = None
    quarter: Optional[int] = None
    filing_type: Optional[str] = None
```

### TimeSeriesParams

The `TimeSeriesParams` model defines parameters for time series queries:

```python
class TimeSeriesParams(BaseModel):
    """Parameters for time series queries."""
    ticker: str
    metric: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: Optional[str] = None
```

### FinancialRatiosParams

The `FinancialRatiosParams` model defines parameters for financial ratios queries:

```python
class FinancialRatiosParams(BaseModel):
    """Parameters for financial ratios queries."""
    ticker: str
    ratios: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
```

## Semantic Search Models

### SemanticSearchParams

The `SemanticSearchParams` model defines parameters for semantic search queries:

```python
class SemanticSearchParams(BaseModel):
    """Parameters for semantic search queries."""
    query: str
    companies: Optional[List[str]] = None
    top_k: int = 5
    filing_types: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None
    sections: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    hybrid_search_weight: float = 0.5
```

## Graph Query Models

### CompanyFilingsParams

The `CompanyFilingsParams` model defines parameters for company filings queries:

```python
class CompanyFilingsParams(BaseModel):
    """Parameters for company filings queries."""
    ticker: str
    filing_types: Optional[List[str]] = None
    limit: int = 10
```

### FilingSectionsParams

The `FilingSectionsParams` model defines parameters for filing sections queries:

```python
class FilingSectionsParams(BaseModel):
    """Parameters for filing sections queries."""
    accession_number: str
    section_types: Optional[List[str]] = None
    limit: int = 50
```

### RelatedCompaniesParams

The `RelatedCompaniesParams` model defines parameters for related companies queries:

```python
class RelatedCompaniesParams(BaseModel):
    """Parameters for related companies queries."""
    ticker: str
    relationship_type: str = "MENTIONS"
    limit: int = 10
```

## Tool Details Models

### ToolDetailsParams

The `ToolDetailsParams` model defines parameters for tool details queries:

```python
class ToolDetailsParams(BaseModel):
    """Parameters for tool details queries."""
    tool_name: str
```

## Custom Query Models

### CustomSQLParams

The `CustomSQLParams` model defines parameters for custom SQL queries:

```python
class CustomSQLParams(BaseModel):
    """Parameters for custom SQL queries."""
    sql_query: str
```

### CustomCypherParams

The `CustomCypherParams` model defines parameters for custom Cypher queries:

```python
class CustomCypherParams(BaseModel):
    """Parameters for custom Cypher queries."""
    cypher_query: str
    query_params: Dict[str, Any] = {}
```

## Validation

The parameter models are used to validate parameters before tool execution:

```python
def validate_call(tool_name: str, query_type: str, params: Dict[str, Any]) -> None:
    """Validate a tool call before execution."""
    # Get the tool spec
    tool_spec = ToolRegistry.get_tool_spec(tool_name)
    if not tool_spec:
        raise ValueError(f"Tool '{tool_name}' not found in registry")
    
    # Check if the query type is supported
    if query_type not in tool_spec.input_schema:
        supported_types = list(tool_spec.input_schema.keys())
        raise QueryTypeUnsupported(query_type, tool_name, supported_types)
    
    # Get the parameter model
    param_model: Type[BaseModel] = tool_spec.input_schema[query_type]
    
    # Validate the parameters
    try:
        param_model(**params)
    except ValidationError as e:
        # Extract field information from the validation error
        field = None
        if e.errors() and 'loc' in e.errors()[0]:
            field = e.errors()[0]['loc'][0]
        
        raise ParameterError(str(e), {'field': field})
```

## Benefits

Using parameter models provides several benefits:

1. **Validation**: Parameters are validated before tool execution, preventing errors
2. **Documentation**: The parameter models provide clear documentation of what each tool expects
3. **Consistency**: All tools have a consistent parameter structure
4. **Error Handling**: Validation errors provide clear error messages
5. **Type Safety**: The parameter models provide type safety for tool parameters
