# Agent Parameters and Phases

This document explains the parameters and phases used in the SEC Filing Analyzer agent architecture.

## Agent Parameters

The agent architecture uses several parameters to control its behavior. These parameters are organized into categories:

### Agent Iteration Parameters

These parameters control how many iterations the agent performs in each phase and are crucial for balancing thoroughness with efficiency:

- **`max_iterations`**: Legacy parameter that used to control the total number of iterations. Still supported for backward compatibility but no longer directly exposed in the configuration. If explicitly set, it overrides the computed effective max iterations.
  - Default: Not explicitly set (derived from phase iterations)
  - Recommended: Don't set directly; configure phase-specific limits instead

- **`max_iterations_effective`**: The actual parameter used to control the total number of iterations. Automatically computed from the phase-specific iteration limits unless `max_iterations` is explicitly set. When `current_iteration >= max_iterations_effective`, the agent terminates regardless of which phase it's in.
  - Computation: `max_iterations if explicitly_set else (sum_of_phase_iterations + small_buffer)`
  - Default: Derived from phase iterations (e.g., 8 for QA Specialist from 1+5+2)
  - The small buffer (10% of the sum, minimum 1) provides extra safety margin

- **`max_planning_iterations`**: Maximum iterations for the planning phase. During planning, the agent analyzes the input, extracts key information, and creates a structured plan. When `phase_iterations['planning'] >= max_planning_iterations`, the agent transitions from planning to execution phase.
  - Default: 2 (set to 1 for QA Specialist)
  - Recommended: 1-2 for most tasks

- **`max_execution_iterations`**: Maximum iterations for the execution phase. During execution, the agent carries out the plan, typically by making tool calls to gather information or perform actions. When `phase_iterations['execution'] >= max_execution_iterations`, the agent transitions from execution to refinement phase.
  - Default: 3 (increased to 5 for QA Specialist)
  - Recommended: 3-5 for most tasks, higher for complex data gathering

- **`max_refinement_iterations`**: Maximum iterations for the refinement phase. During refinement, the agent processes all gathered information to produce a final, polished result. When `phase_iterations['refinement'] >= max_refinement_iterations`, the agent terminates.
  - Default: 1 (increased to 2 for QA Specialist)
  - Recommended: 1-2 for most tasks

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

## Agent Loop and Phase Transitions

### Agent Loop Structure

The agent loop follows a three-phase structure, with each phase having its own iteration counter and maximum limit:

```
Agent Loop
├── Planning Phase (max_planning_iterations)
│   └── Iterations: 0, 1, 2, ...
├── Execution Phase (max_execution_iterations)
│   └── Iterations: 0, 1, 2, ...
└── Refinement Phase (max_refinement_iterations)
    └── Iterations: 0, 1, 2, ...
```

The overall `max_iterations` parameter limits the total number of iterations across all phases combined.

### Phase Transitions

The agent transitions between phases based on the following criteria:

1. **Planning → Execution**: When `phase_iterations['planning'] >= max_planning_iterations`
2. **Execution → Refinement**: When `phase_iterations['execution'] >= max_execution_iterations` or when execution is complete (e.g., all agents have run in the Coordinator)
3. **Refinement → Termination**: When `phase_iterations['refinement'] >= max_refinement_iterations`

### Iteration Counting

Each phase has its own iteration counter in `state.phase_iterations`, which is incremented at the end of each iteration in that phase. The overall iteration counter `state.current_iteration` is incremented regardless of which phase the agent is in.

### Termination Conditions

The agent terminates under any of the following conditions:

1. **Max Iterations**: When `state.current_iteration >= max_iterations`
2. **Phase Completion**: When all phases have completed their maximum iterations
3. **Dynamic Termination**: When `enable_dynamic_termination` is true and the agent determines it has a high-confidence answer
4. **Max Duration**: When `time.time() - state.start_time >= max_duration_seconds`
5. **Capability Termination**: When a capability signals that the agent should terminate

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

### Setting Iteration Parameters

1. **Balance Across Phases**: Allocate iterations strategically across phases based on task complexity:
   - For simple, factual queries: 1 planning, 2-3 execution, 1 refinement
   - For complex analysis: 1-2 planning, 4-5 execution, 2 refinement

2. **Adjust for Task Type**:
   - **QA Specialist**: Higher execution iterations (5+) for complex financial queries
   - **Financial Analyst**: Higher execution iterations (4-5) for detailed financial analysis
   - **Risk Analyst**: Balanced iterations across phases (2 planning, 3-4 execution, 2 refinement)
   - **Coordinator**: Higher planning iterations (2-3) for complex orchestration

3. **Consider Tool Complexity**:
   - Tasks requiring multiple tool calls need higher execution iterations
   - Tasks with complex data processing need higher refinement iterations

4. **Global vs. Phase Limits**:
   - Focus on setting the phase-specific iteration limits (`max_planning_iterations`, `max_execution_iterations`, `max_refinement_iterations`)
   - Let the system automatically compute `max_iterations_effective` from these phase limits
   - Only override `max_iterations` if you have specific requirements
   - The system adds a small buffer (10% of the sum, minimum 1) to the sum of phase iterations for safety

5. **Monitor and Adjust**:
   - Review agent logs to see if iterations are being fully utilized
   - If the agent consistently terminates early, reduce iterations
   - If the agent consistently hits iteration limits, increase them

### Other Best Practices

1. **Tool Retries**: Set to 1-2 for most tools, higher for less reliable tools.
2. **Tools Per Iteration**: Keep at 1 to avoid parameter confusion, unless tools are very simple.
3. **Dynamic Termination**: Enable for tasks where quality can vary, disable for critical tasks.
4. **Circuit Breaker**: Use default settings unless you have specific reliability issues with certain tools.
