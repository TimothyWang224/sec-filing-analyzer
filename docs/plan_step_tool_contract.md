# Plan-Step ↔ Tool Contract

This document describes the contract between plan steps and tools in the SEC Filing Analyzer system.

## Overview

The Plan-Step ↔ Tool Contract defines a clear interface between planning steps and the tools they use. It ensures that:

1. Plan steps have a consistent structure with well-defined fields
2. Tools have a consistent interface with well-defined input and output schemas
3. Results from tools are stored in memory with consistent keys
4. Steps can be skipped based on success criteria

## Models

### ToolSpec

The `ToolSpec` model defines the contract for what a tool expects and returns:

```python
class ToolSpec(BaseModel):
    """Tool specification."""
    name: str
    input_schema: Dict[str, Any]
    output_key: str
    description: str = ""
```

- `name`: The name of the tool
- `input_schema`: The schema for the tool's input parameters
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
4. The tool returns a result with an `output_key` field
5. The agent stores the result in memory using the `expected_key` from the step

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

## Benefits

The Plan-Step ↔ Tool Contract provides several benefits:

1. **Consistency**: All plan steps and tools have a consistent structure
2. **Clarity**: The contract clearly defines what each step expects and provides
3. **Flexibility**: Steps can be skipped based on success criteria
4. **Reusability**: Tools can be reused across different plans
5. **Maintainability**: The contract makes it easier to maintain and extend the system
