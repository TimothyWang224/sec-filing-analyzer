# Agent Loop and Iteration Parameters

This document provides a detailed explanation of the agent loop and iteration parameters in the SEC Filing Analyzer.

## Agent Loop Overview

The agent loop is the core execution mechanism for all agents in the system. It follows a three-phase structure:

1. **Planning Phase**: The agent analyzes the input and creates a plan
2. **Execution Phase**: The agent executes the plan, typically by making tool calls
3. **Refinement Phase**: The agent refines the results into a final output

Each phase has its own iteration counter and maximum limit, and the overall loop has a global iteration limit.

## Iteration Parameters

### `max_iterations` and `max_iterations_effective`

#### `max_iterations` (Legacy Parameter)

**Description**: This is the legacy parameter that used to control the total number of iterations across all phases. It's still supported for backward compatibility but is no longer directly exposed in the configuration.

**Behavior**: If explicitly set, it overrides the computed effective max iterations.

#### `max_iterations_effective` (Derived Parameter)

**Description**: This is the actual parameter used to control the total number of iterations. It's automatically computed from the phase-specific iteration limits unless `max_iterations` is explicitly set.

**Computation**:
```python
max_iterations_effective = max_iterations if explicitly_set else (
    max_planning_iterations +
    max_execution_iterations +
    max_refinement_iterations +
    small_buffer
)
```

The small buffer (10% of the sum, minimum 1) provides extra safety margin for phase transitions or other edge cases.

**Behavior**: When `current_iteration >= max_iterations_effective`, the agent terminates regardless of which phase it's in.

**Default Values**:
- Computed from phase iterations with a small buffer
- QA Specialist: 8 (derived from 1 planning + 5 execution + 2 refinement)
- Coordinator: 6 (derived from 2 planning + 3 execution + 1 refinement)
- Financial Analyst: 6 (derived from 2 planning + 3 execution + 1 refinement)
- Risk Analyst: 6 (derived from 2 planning + 3 execution + 1 refinement)

**Recommended Approach**:
- Don't set `max_iterations` directly; instead, configure the phase-specific iteration limits
- Let the system compute `max_iterations_effective` automatically
- Only override `max_iterations` if you have specific requirements

### `max_planning_iterations`

**Description**: Maximum iterations for the planning phase. During planning, the agent analyzes the input, extracts key information, and creates a structured plan.

**Behavior**: When `phase_iterations['planning'] >= max_planning_iterations`, the agent transitions from planning to execution phase.

**Default Values**:
- Base Agent: 2
- QA Specialist: 1
- Coordinator: 2
- Financial Analyst: 2
- Risk Analyst: 2

**Recommended Values**:
- Most tasks: 1-2
- Complex planning tasks: 2-3

### `max_execution_iterations`

**Description**: Maximum iterations for the execution phase. During execution, the agent carries out the plan, typically by making tool calls to gather information or perform actions.

**Behavior**: When `phase_iterations['execution'] >= max_execution_iterations`, the agent transitions from execution to refinement phase.

**Default Values**:
- Base Agent: 3
- QA Specialist: 5 (increased to allow more tool calls)
- Coordinator: 3
- Financial Analyst: 3
- Risk Analyst: 3

**Recommended Values**:
- Simple data gathering: 2-3
- Complex data gathering: 4-5
- Very complex data gathering: 5-10

### `max_refinement_iterations`

**Description**: Maximum iterations for the refinement phase. During refinement, the agent processes all gathered information to produce a final, polished result.

**Behavior**: When `phase_iterations['refinement'] >= max_refinement_iterations`, the agent terminates.

**Default Values**:
- Base Agent: 1
- QA Specialist: 2 (increased for better answer refinement)
- Coordinator: 1
- Financial Analyst: 1
- Risk Analyst: 1

**Recommended Values**:
- Simple refinement: 1
- Complex refinement: 2-3

## Agent Loop Execution Flow

The agent loop follows this execution flow:

```
Initialize agent state
  ↓
Start in Planning Phase
  ↓
While not terminated:
  ↓
  If in Planning Phase:
    → Execute planning logic
    → Increment planning iteration counter
    → If planning iterations >= max_planning_iterations:
        → Transition to Execution Phase
  ↓
  If in Execution Phase:
    → Execute execution logic (typically tool calls)
    → Increment execution iteration counter
    → If execution iterations >= max_execution_iterations:
        → Transition to Refinement Phase
  ↓
  If in Refinement Phase:
    → Execute refinement logic
    → Increment refinement iteration counter
    → If refinement iterations >= max_refinement_iterations:
        → Terminate
  ↓
  Increment overall iteration counter
  ↓
  Check termination conditions:
    → If overall iterations >= max_iterations: Terminate
    → If max duration exceeded: Terminate
    → If dynamic termination enabled and confidence high: Terminate
```

## Iteration Counting

Each phase has its own iteration counter in `state.phase_iterations`, which is incremented at the end of each iteration in that phase. The overall iteration counter `state.current_iteration` is incremented regardless of which phase the agent is in.

```python
def increment_iteration(self) -> None:
    """Increment the current iteration counter."""
    self.current_iteration += 1
    self.phase_iterations[self.current_phase] += 1
```

## Termination Conditions

The agent terminates under any of the following conditions:

1. **Max Iterations**: When `state.current_iteration >= max_iterations`
2. **Phase Completion**: When all phases have completed their maximum iterations
3. **Dynamic Termination**: When `enable_dynamic_termination` is true and the agent determines it has a high-confidence answer
4. **Max Duration**: When `time.time() - state.start_time >= max_duration_seconds`
5. **Capability Termination**: When a capability signals that the agent should terminate

## Balancing Iterations Across Phases

When setting iteration parameters, it's important to balance iterations across phases based on the task complexity:

1. **Planning Phase**: Usually needs fewer iterations (1-2) since it's primarily about understanding the task and creating a plan.

2. **Execution Phase**: Usually needs more iterations (3-5) since this is where most of the work happens, including tool calls and data gathering.

3. **Refinement Phase**: Usually needs fewer iterations (1-2) since it's primarily about improving the quality of the answer.

The sum of phase iterations should generally be less than or equal to `max_iterations` to ensure the agent has enough iterations to complete all phases.

## Example Configurations

### Simple Factual Query

```python
agent_config = {
    "max_iterations": 5,
    "max_planning_iterations": 1,
    "max_execution_iterations": 3,
    "max_refinement_iterations": 1
}
```

### Complex Financial Analysis

```python
agent_config = {
    "max_iterations": 10,
    "max_planning_iterations": 2,
    "max_execution_iterations": 5,
    "max_refinement_iterations": 2
}
```

### Multi-Agent Coordination

```python
coordinator_config = {
    "max_iterations": 15,
    "max_planning_iterations": 3,
    "max_execution_iterations": 8,
    "max_refinement_iterations": 3
}
```

## Monitoring and Adjusting

It's important to monitor agent logs to see if iterations are being fully utilized:

- If the agent consistently terminates early, reduce iterations to improve efficiency
- If the agent consistently hits iteration limits, increase them to improve thoroughness
- If the agent spends too much time in one phase, adjust the balance of iterations across phases

## Conclusion

The agent loop and iteration parameters are crucial for controlling the behavior of agents in the SEC Filing Analyzer. By understanding how these parameters work and how to set them appropriately, you can optimize agent performance for different types of tasks.
