# Agent Parameters and Phases

This document explains the parameters and phases used in the SEC Filing Analyzer agent architecture.

## Agent Parameters

The agent architecture uses several parameters to control its behavior. These parameters are organized into categories:

### Agent Iteration Parameters

These parameters control how many iterations the agent performs in each phase:

- **`max_iterations`**: Legacy parameter for backward compatibility. Controls the total number of iterations across all phases.
- **`max_planning_iterations`**: Maximum iterations for the planning phase. Default: 1
- **`max_execution_iterations`**: Maximum iterations for the execution phase. Default: 2
- **`max_refinement_iterations`**: Maximum iterations for result refinement. Default: 1

### Tool Execution Parameters

These parameters control how tools are executed:

- **`max_tool_retries`**: Number of times to retry a failed tool call before reporting failure. Default: 2
- **`tools_per_iteration`**: Number of tools to execute per iteration. Default: 1 (single tool call approach)
- **`circuit_breaker_threshold`**: Number of consecutive failures before opening the circuit. Default: 3
- **`circuit_breaker_reset_timeout`**: Time in seconds before attempting to reset the circuit. Default: 300

### Runtime Parameters

These parameters control the overall runtime of the agent:

- **`max_duration_seconds`**: Maximum runtime in seconds. Default: 180

### Termination Parameters

These parameters control when the agent should terminate:

- **`enable_dynamic_termination`**: Whether to allow early termination based on result quality. Default: false
- **`min_confidence_threshold`**: Minimum confidence score to consider a result satisfactory. Default: 0.8

## Agent Phases

The agent operates in three distinct phases:

### 1. Planning Phase

In the planning phase, the agent focuses on understanding the task and creating a detailed plan:

- Analyzes the question or task
- Identifies key information needed
- Creates a step-by-step plan for execution
- Determines which tools and parameters will be needed

### 2. Execution Phase

In the execution phase, the agent focuses on gathering data and generating an initial answer:

- Executes the plan created in the planning phase
- Calls appropriate tools to gather information
- Processes and analyzes the collected data
- Generates an initial answer or result

### 3. Refinement Phase

In the refinement phase, the agent focuses on improving the quality of the answer:

- Reviews the initial answer for accuracy and completeness
- Makes the answer more concise and directly responsive to the question
- Improves explanations of complex concepts
- Ensures numerical data is presented clearly
- Adds confidence scores if dynamic termination is enabled

## Phase Transitions

The agent transitions between phases based on the following criteria:

1. **Planning → Execution**: When `phase_iterations['planning'] >= max_planning_iterations`
2. **Execution → Refinement**: When `phase_iterations['execution'] >= max_execution_iterations`
3. **Refinement → Termination**: When `phase_iterations['refinement'] >= max_refinement_iterations`

## Tool Ledger

The agent uses a Tool Ledger to track tool calls and their results:

- Records all tool calls, their parameters, and results
- Maintains a chronological record for reference
- Formats previous tool results for inclusion in prompts
- Tracks retry information for failed tool calls

## Error Handling

The agent uses a comprehensive error handling system to recover from tool call failures:

### Error Classification

Errors are classified into different types based on their cause:

- **Parameter Error**: Invalid or missing parameters
- **Network Error**: Network connectivity issues
- **Authentication Error**: Authentication/authorization issues
- **Rate Limit Error**: Rate limiting or quota issues
- **Data Error**: Data not found or invalid data
- **System Error**: Internal system errors
- **Unknown Error**: Unclassified errors

### Recovery Strategies

Different recovery strategies are applied based on the error type:

- **Parameter Fixing**: Attempts to fix parameter errors using the LLM
- **Adaptive Retry**: Retries failed tool calls with appropriate backoff strategies
- **Circuit Breaking**: Prevents repeated calls to failing tools
- **Alternative Tools**: Suggests alternative tools when a primary tool fails

### User Feedback

The error handling system provides user-friendly error messages and suggestions for fixing common errors. This helps users understand and resolve issues more effectively.

For more details, see the [Error Handling](error_handling.md) documentation.

## Example Configuration

```python
agent = QASpecialistAgent(
    # Agent iteration parameters
    max_planning_iterations=1,
    max_execution_iterations=2,
    max_refinement_iterations=1,

    # Tool execution parameters
    max_tool_retries=2,
    tools_per_iteration=1,
    circuit_breaker_threshold=3,
    circuit_breaker_reset_timeout=300,

    # Runtime parameters
    max_duration_seconds=180,

    # Termination parameters
    enable_dynamic_termination=True,
    min_confidence_threshold=0.8
)
```

## Best Practices

1. **Planning Phase**: Keep this phase short (1-2 iterations) to focus on understanding the task.
2. **Execution Phase**: Allow more iterations (2-3) for data gathering and initial answer generation.
3. **Refinement Phase**: Use 1-2 iterations for improving answer quality.
4. **Tool Retries**: Set to 1-2 for most tools, higher for less reliable tools.
5. **Tools Per Iteration**: Keep at 1 to avoid parameter confusion, unless tools are very simple.
6. **Dynamic Termination**: Enable for tasks where quality can vary, disable for critical tasks.
