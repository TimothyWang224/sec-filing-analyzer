# Planning Capability

The Planning Capability is a component of the SEC Filing Analyzer agent architecture that enables agents to create, manage, and execute plans for complex tasks.

## Overview

The Planning Capability allows agents to break down complex tasks into manageable steps, track progress, and adapt plans as needed. It's particularly useful for coordinator agents that need to orchestrate multiple subtasks.

## Features

- **Plan Generation**: Creates detailed plans based on user input
- **Step Tracking**: Tracks the current step and completed steps
- **Dynamic Replanning**: Updates plans based on new information
- **Step Reflection**: Reflects on step results to improve future steps
- **Phase Awareness**: Integrates with the agent's phase-based execution model

## Implementation

The Planning Capability is implemented as a class in `src/capabilities/planning.py`. It extends the base `Capability` class and provides methods for:

- **Plan Creation**: `_create_plan()`
- **Plan Execution**: `process_action()`
- **Plan Reflection**: `_reflect_and_update_plan()`
- **Prompt Enhancement**: `process_prompt()`

## Usage

### Adding the Planning Capability to an Agent

```python
from src.capabilities.planning import PlanningCapability

# Create a planning capability
planning = PlanningCapability(
    enable_dynamic_replanning=True,
    enable_step_reflection=True,
    min_steps_before_reflection=2,
    max_plan_steps=10,
    plan_detail_level="high"
)

# Add it to an agent
agent = QASpecialistAgent(
    capabilities=[planning],
    max_planning_iterations=1,
    max_execution_iterations=2,
    max_refinement_iterations=1
)
```

### Plan Structure

Plans created by the Planning Capability have the following structure:

```json
{
  "goal": "Analyze financial information based on: What was Apple's revenue in 2023?",
  "steps": [
    {
      "step_id": 1,
      "description": "Retrieve Apple's financial data for 2023",
      "tool": "sec_financial_data",
      "parameters": {
        "query_type": "financial_facts",
        "parameters": {
          "ticker": "AAPL",
          "metrics": ["Revenue"],
          "start_date": "2023-01-01",
          "end_date": "2023-12-31"
        }
      },
      "dependencies": [],
      "status": "pending"
    },
    {
      "step_id": 2,
      "description": "Search for relevant context in SEC filings",
      "tool": "sec_semantic_search",
      "parameters": {
        "query": "Apple revenue 2023",
        "companies": ["AAPL"],
        "filing_types": ["10-K", "10-Q"],
        "date_range": ["2023-01-01", "2023-12-31"]
      },
      "dependencies": [],
      "status": "pending"
    },
    {
      "step_id": 3,
      "description": "Generate comprehensive answer",
      "dependencies": [1, 2],
      "status": "pending"
    }
  ],
  "status": "in_progress",
  "created_at": "2025-04-15T12:34:56.789012"
}
```

## Integration with Phase-Based Execution

The Planning Capability has been enhanced to work with the agent's phase-based execution model:

1. **Planning Phase**: Creates a plan and identifies the information needed
2. **Execution Phase**: Executes the plan steps to gather data
3. **Refinement Phase**: Refines the results based on the gathered data

The capability adds phase-specific guidance to the prompt:

- **Planning Phase**: Focus on understanding the task and creating a detailed plan
- **Execution Phase**: Focus on gathering data and generating an initial answer
- **Refinement Phase**: Focus on improving the answer quality

## Benefits

The Planning Capability provides several benefits:

1. **Structured Approach**: Breaks down complex tasks into manageable steps
2. **Better Tool Usage**: Recommends appropriate tools and parameters for each step
3. **Progress Tracking**: Tracks progress through the plan
4. **Adaptability**: Updates plans based on new information
5. **Phase Integration**: Works seamlessly with the agent's phase-based execution model

## Example

Here's an example of how the Planning Capability enhances the agent's prompt:

```
Current Phase: EXECUTION

Current Plan Step (2/3):
- Description: Search for relevant context in SEC filings
- Recommended Tool: sec_semantic_search

In the EXECUTION phase, focus on gathering data and generating an initial answer.
Use the appropriate tools to collect the necessary information.

Previous Tool Calls:

--- Call 1 ---
Tool: sec_financial_data
Args: {
  "query_type": "financial_facts",
  "parameters": {
    "ticker": "AAPL",
    "metrics": ["Revenue"],
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
  }
}
Result: {"revenue": 123456789, "net_income": 12345678}
```
