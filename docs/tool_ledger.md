# Tool Ledger

The Tool Ledger is a component of the SEC Filing Analyzer agent architecture that tracks tool calls and their results.

## Overview

The Tool Ledger maintains a chronological record of all tool calls, their parameters, and their results or errors. This makes it easier for agents to reference previous tool calls and build on their results.

## Features

- **Comprehensive Record**: Tracks all tool calls, their parameters, and results
- **Chronological Order**: Maintains a chronological record for reference
- **Formatting**: Formats previous tool results for inclusion in prompts
- **Retry Tracking**: Tracks retry information for failed tool calls
- **Memory Integration**: Integrates with agent memory for persistent storage

## Implementation

The Tool Ledger is implemented as a class in `src/agents/core/tool_ledger.py`. It provides methods for:

- **Recording Tool Calls**: `record_tool_call()`
- **Retrieving Entries**: `get_entries()`, `get_latest_entry()`, `get_entry_by_id()`
- **Formatting for Prompts**: `format_for_prompt()`
- **Converting to Memory Format**: `to_memory_format()`

## Usage

### Recording a Tool Call

```python
# Record a successful tool call
ledger.record_tool_call(
    tool_name="sec_financial_data",
    args={"query_type": "financial_facts", "parameters": {"ticker": "AAPL"}},
    result={"revenue": 123456789},
    metadata={"duration": 0.5}
)

# Record a failed tool call
ledger.record_tool_call(
    tool_name="sec_financial_data",
    args={"query_type": "financial_facts", "parameters": {"ticker": "INVALID"}},
    error="Company not found",
    metadata={"duration": 0.2, "retries": 2}
)
```

### Retrieving Entries

```python
# Get all entries
all_entries = ledger.entries

# Get the latest entry
latest_entry = ledger.get_latest_entry()

# Get entries for a specific tool
financial_data_entries = ledger.get_entries(tool_name="sec_financial_data")

# Get successful entries
successful_entries = ledger.get_entries(status="success")

# Get failed entries
failed_entries = ledger.get_entries(status="error")
```

### Formatting for Prompts

```python
# Format the last 3 entries for inclusion in a prompt
formatted_entries = ledger.format_for_prompt(limit=3)
```

### Converting to Memory Format

```python
# Convert entries to memory format
memory_items = ledger.to_memory_format()
```

## Integration with Agents

The Tool Ledger is integrated with the agent architecture in several ways:

1. **Agent Base Class**: The `Agent` class initializes a Tool Ledger in its constructor
2. **Tool Execution**: The `execute_tool_calls` method records tool calls in the ledger
3. **Planning Capability**: The `PlanningCapability` includes ledger entries in prompts
4. **Memory Integration**: Tool results are stored in both the ledger and agent memory

## Benefits

The Tool Ledger provides several benefits:

1. **Improved Context**: Agents have better context about previous tool calls
2. **Reduced Redundancy**: Agents can avoid repeating the same tool calls
3. **Better Error Handling**: Failed tool calls are tracked for better error recovery
4. **Enhanced Debugging**: The ledger provides a comprehensive record for debugging

## Example

Here's an example of how the Tool Ledger is used in the agent architecture:

```python
# Agent executes a tool call
result = await agent.execute_tool_calls([
    {
        "tool": "sec_financial_data",
        "args": {
            "query_type": "financial_facts",
            "parameters": {
                "ticker": "AAPL"
            }
        }
    }
])

# The tool call is recorded in the ledger
print(agent.tool_ledger.format_for_prompt())
```

This will output something like:

```
Previous Tool Calls:

--- Call 1 ---
Tool: sec_financial_data
Args: {
  "query_type": "financial_facts",
  "parameters": {
    "ticker": "AAPL"
  }
}
Result: {"revenue": 123456789, "net_income": 12345678}
```
