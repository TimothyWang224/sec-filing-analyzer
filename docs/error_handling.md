# Enhanced Error Handling and Recovery

This document describes the enhanced error handling and recovery capabilities of the agent system.

## Overview

The enhanced error handling system provides a comprehensive approach to handling errors in agent tool calls. It includes:

1. **Error Classification**: Classifies errors into different types based on their cause
2. **Adaptive Retry Strategy**: Implements different retry strategies based on error type
3. **Circuit Breaker Pattern**: Prevents repeated calls to failing tools
4. **Alternative Tool Selection**: Suggests alternative tools when a primary tool fails
5. **Error Analysis**: Analyzes patterns of errors to improve future tool calls
6. **Enhanced User Feedback**: Provides user-friendly error messages

## Components

### Error Classification

The `ErrorClassifier` class classifies exceptions into different error types:

- **Parameter Error**: Invalid or missing parameters
- **Network Error**: Network connectivity issues
- **Authentication Error**: Authentication/authorization issues
- **Rate Limit Error**: Rate limiting or quota issues
- **Data Error**: Data not found or invalid data
- **System Error**: Internal system errors
- **Unknown Error**: Unclassified errors

Each error type has a recommended recovery strategy and a determination of whether it's potentially recoverable.

### Adaptive Retry Strategy

The `AdaptiveRetryStrategy` class implements different retry strategies based on error type:

- **Exponential Backoff**: For network errors
- **Longer Backoff**: For rate limit errors
- **Default Backoff**: For other recoverable errors

The retry strategy includes jitter to prevent thundering herd problems and respects a maximum delay to prevent excessive waiting.

### Circuit Breaker Pattern

The `ToolCircuitBreaker` class implements the circuit breaker pattern to prevent repeated calls to failing tools:

- **Closed State**: Tool calls are allowed
- **Open State**: Tool calls are blocked
- **Reset Timeout**: After a timeout, a single test call is allowed to check if the issue is resolved

The circuit breaker tracks failure counts for each tool and opens the circuit when a threshold is reached.

### Alternative Tool Selection

The `AlternativeToolSelector` class suggests alternative tools when a primary tool fails:

- **Tool Selection**: Uses an LLM to suggest alternative tools that can achieve the same purpose
- **Parameter Mapping**: Maps parameters from the failed tool to the alternative tool

This allows the agent to continue its task even when a specific tool is unavailable.

### Error Analysis

The `ErrorAnalyzer` class analyzes patterns of errors to improve future tool calls:

- **Error History**: Tracks the history of errors for each tool
- **Common Errors**: Identifies the most common error types for each tool
- **Error Suggestions**: Provides suggestions for fixing common errors

This helps the agent learn from past errors and improve its tool usage over time.

### Comprehensive Error Recovery

The `ErrorRecoveryManager` class combines all these components into a comprehensive error recovery system:

1. **Parameter Fixing**: Attempts to fix parameter errors using the LLM
2. **Adaptive Retry**: Retries failed tool calls with appropriate backoff
3. **Circuit Breaking**: Prevents repeated calls to failing tools
4. **Alternative Tools**: Suggests alternative tools when a primary tool fails
5. **User Feedback**: Provides user-friendly error messages

## Integration with Agents

The error handling system is integrated with the agent architecture in several ways:

1. **Agent Initialization**: The `Agent` class initializes an `ErrorRecoveryManager` instance
2. **Tool Execution**: The `execute_tool_calls` method uses the error recovery manager to execute tool calls
3. **Error Reporting**: Tool errors are recorded in the agent's memory and tool ledger
4. **User Feedback**: User-friendly error messages are included in the agent's response

## Example

Here's an example of how the error handling system works:

```python
# Define a function to execute the tool
async def execute_tool(tool_name, args):
    return await self.environment.execute_action({
        "tool": tool_name,
        "args": args
    })

# Execute the tool with comprehensive error recovery
recovery_result = await self.error_recovery_manager.execute_tool_with_recovery(
    tool_name=tool_name,
    tool_args=tool_args,
    execute_func=execute_tool,
    user_input=user_input,
    context=self.state.get_context()
)

# Process the result based on success/failure
if recovery_result["success"]:
    # Handle successful result
    result = recovery_result["result"]
    
    # Check if recovery was needed
    if "recovery_strategy" in recovery_result:
        # Recovery was successful
        strategy = recovery_result["recovery_strategy"]
        # ...
else:
    # Handle error
    error = recovery_result["error"]
    error_message = error.message
    error_type = error.error_type.value
    
    # Get user-friendly error message
    user_message = recovery_result.get("user_message", error_message)
    
    # Get suggestions for fixing the error
    suggestions = recovery_result.get("suggestions", [])
    # ...
```

## Benefits

The enhanced error handling system provides several benefits:

1. **Improved Reliability**: The agent can recover from many types of errors
2. **Better User Experience**: User-friendly error messages help users understand and fix issues
3. **Reduced Downtime**: Circuit breaking prevents cascading failures
4. **Adaptive Recovery**: Different recovery strategies for different error types
5. **Learning from Errors**: Error analysis helps improve future tool calls

## Testing

A test script is provided to demonstrate the enhanced error handling capabilities:

```bash
python scripts/tests/test_error_recovery.py
```

This script tests the error classifier, adaptive retry strategy, and comprehensive error recovery manager with various error scenarios.
