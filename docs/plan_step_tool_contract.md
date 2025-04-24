# Plan-Step ↔ Tool Contract

This document describes the contract between plan steps and tools in the SEC Filing Analyzer system.

## Overview

The Plan-Step ↔ Tool Contract defines a clear interface between planning steps and the tools they use. It ensures that:

1. Plan steps have a consistent structure with well-defined fields
2. Tools have a consistent interface with well-defined input and output schemas
3. Results from tools are stored in memory with consistent keys
4. Steps can be skipped based on success criteria
5. Parameters are validated before tool execution
6. Errors are properly classified and handled with user-friendly messages

## Models

### ToolSpec

The `ToolSpec` model defines the contract for what a tool expects and returns:

```python
class ToolSpec(BaseModel):
    """Tool specification."""
    name: str
    input_schema: Dict[str, Type[BaseModel]]  # query_type -> parameter model
    output_key: str
    description: str = ""
```

- `name`: The name of the tool
- `input_schema`: A mapping from query types to Pydantic parameter models
- `output_key`: The key to use when storing the tool's output in memory
- `description`: A description of the tool

### PlanStep

The `PlanStep` model defines a step in a plan with a clear contract for what it expects and provides:

```python
class PlanStep(BaseModel):
    """Plan step."""
    step_id: int
    description: str
    tool: Optional[str] = None
    agent: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[int] = Field(default_factory=list)
    expected_key: Optional[str] = None
    output_path: Optional[List[str]] = None
    done_check: Optional[str] = None
    status: str = "pending"
    completed_at: Optional[str] = None
    skipped: bool = False
```

- `step_id`: The unique identifier for the step
- `description`: A description of the step
- `tool`: The name of the tool to use (if any)
- `agent`: The name of the agent to use (if any)
- `parameters`: The parameters to pass to the tool or agent
- `dependencies`: The IDs of steps that must be completed before this step
- `expected_key`: The key to use when storing the step's output in memory
- `output_path`: The path to the output value in the tool's result
- `done_check`: A condition to check if the step is already done
- `status`: The status of the step (pending, in_progress, completed, failed)
- `completed_at`: The timestamp when the step was completed
- `skipped`: Whether the step was skipped

### Plan

The `Plan` model defines a complete plan with steps and metadata:

```python
class Plan(BaseModel):
    """Plan."""
    goal: str
    steps: List[PlanStep] = Field(default_factory=list)
    status: str = "pending"
    created_at: str
    completed_at: Optional[str] = None
    owner: str = "agent"
    can_modify: bool = True
```

- `goal`: The goal of the plan
- `steps`: The steps in the plan
- `status`: The status of the plan (pending, in_progress, completed, failed)
- `created_at`: The timestamp when the plan was created
- `completed_at`: The timestamp when the plan was completed
- `owner`: The owner of the plan
- `can_modify`: Whether the plan can be modified

## Workflow

### Creating a Plan

1. The `PlanningCapability` creates a plan based on the user input
2. The plan is converted to a `Plan` object with `PlanStep` objects
3. The plan is stored in the agent's memory

### Executing a Plan

1. The agent executes each step in the plan
2. Before executing a step, the agent checks if the step can be skipped
3. If the step can be skipped, the agent adds a memory item for the skipped step
4. If the step cannot be skipped, the agent executes the step
5. The result of the step is stored in memory using the `expected_key` from the step

### Skipping Steps

A step can be skipped if:

1. The step has a `done_check` condition
2. The condition evaluates to `True` when checked against the agent's memory

When a step is skipped:

1. The agent adds a memory item with `type: "step_skipped"`
2. The agent stores the result in memory using the `expected_key` from the step
3. The agent continues to the next step

### Tool Execution

When a tool is executed:

1. The agent passes the parameters from the step to the tool
2. The tool validates the parameters
3. The tool executes with the validated parameters
4. The tool returns a standardized response with the following fields:
   - `query_type`: The type of query that was executed
   - `parameters`: The parameters that were used
   - `results`: The results of the query (empty list for errors)
   - `output_key`: The tool's name
   - `success`: Boolean indicating whether the operation was successful
   - Additional fields specific to the tool
5. If the response has `success: false`, it will also include:
   - `error` or `warning`: The error message (depending on error type)
6. The agent stores the result in memory using the `expected_key` from the step

All tools use the `format_success_response` and `format_error_response` methods from the `Tool` base class to ensure consistent response formatting.

## Example

Here's an example of a plan step that uses the `sec_financial_data` tool:

```python
step = PlanStep(
    step_id=1,
    description="Get Apple's revenue for 2022",
    tool="sec_financial_data",
    parameters={
        "query_type": "financial_facts",
        "parameters": {
            "ticker": "AAPL",
            "metrics": ["Revenue"],
            "start_date": "2022-01-01",
            "end_date": "2022-12-31"
        }
    },
    expected_key="revenue_data",
    output_path=["data", "Revenue"],
    done_check="True",
    dependencies=[],
    status="pending"
)
```

When this step is executed, the `sec_financial_data` tool is called with the parameters, and the result is stored in memory with the key `revenue_data`. The `output_path` field specifies that the value to extract is at `["data", "Revenue"]` in the result.

## Parameter Models

The Plan-Step ↔ Tool Contract uses Pydantic models to define and validate tool parameters. Each tool defines a set of parameter models for different query types:

```python
class ToolInput(BaseModel):
    """Base model for tool inputs."""
    query_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


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

## Error Handling

The Plan-Step ↔ Tool Contract includes a comprehensive error hierarchy for better error handling:

```python
class ToolError(Exception):
    """Base class for all tool-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def user_message(self) -> str:
        """Return a user-friendly error message."""
        return self.message


class ParameterError(ToolError):
    """Error raised when tool parameters are invalid."""

    def user_message(self) -> str:
        """Return a user-friendly error message."""
        if 'field' in self.details:
            return f"The parameter '{self.details['field']}' is invalid: {self.message}"
        return f"Invalid parameter: {self.message}"


class QueryTypeUnsupported(ToolError):
    """Error raised when a query type is not supported by a tool."""

    def __init__(self, query_type: str, tool_name: str, supported_types: Optional[List[str]] = None):
        self.query_type = query_type
        self.tool_name = tool_name
        self.supported_types = supported_types or []

        message = f"Query type '{query_type}' is not supported by the {tool_name} tool."
        if self.supported_types:
            message += f" Supported types are: {', '.join(self.supported_types)}"

        super().__init__(message, {
            'query_type': query_type,
            'tool_name': tool_name,
            'supported_types': self.supported_types
        })
```

## Validation

The Plan-Step ↔ Tool Contract includes a validation framework for validating tool calls before execution:

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

The Plan-Step ↔ Tool Contract provides several benefits:

1. **Consistency**: All plan steps and tools have a consistent structure
2. **Clarity**: The contract clearly defines what each step expects and provides
3. **Flexibility**: Steps can be skipped based on success criteria
4. **Reusability**: Tools can be reused across different plans
5. **Maintainability**: The contract makes it easier to maintain and extend the system
6. **Validation**: Parameters are validated before tool execution, preventing errors
7. **Error Handling**: Errors are properly classified and handled with user-friendly messages
8. **Documentation**: The parameter models provide clear documentation of what each tool expects
