# SEC Filing Analyzer - To-Do List

## ETL & Storage
1. Storage Synchronization - link to refresh button, script to execute once an ETL process is complete

## QA Agent Optimizations

### ✅ 1. Make Tool Calls Idempotent & Cacheable

```python
from functools import lru_cache, wraps
def memoize_tool(func):
    @lru_cache(maxsize=128)
    def _cached(*a, **kw): return func(*a, **kw)
    @wraps(func)
    def wrapper(*a, **kw): return _cached(*a, **kw)
    return wrapper

@memoize_tool
def sec_financial_data(query_type: str, parameters: dict): ...
```

**Impact**: High - This would immediately prevent redundant tool calls, saving time and resources.
**Difficulty**: Low - The implementation is straightforward with Python's built-in decorators.
**Priority**: High - This is a quick win that would have immediate benefits.

**Result**: Iterations 2-5 reuse the first successful response at zero token cost.

**Implementation**: Created a memoization decorator that caches tool results based on their input parameters. Applied the decorator to the `execute` method in the `Tool` base class to ensure all tool calls are cached. Added a mechanism to clear the cache when needed.

### ✅ 2. Success-Criterion Short-Circuit

```python
if "NetIncome" in memory and memory["NetIncome"]["year"] == 2023:
    phase_done = True           # jump to refinement immediately
```

**Impact**: High - This would allow the agent to move to refinement as soon as it has the information it needs.
**Difficulty**: Medium - Requires modifying the agent's execution logic and defining success criteria for different query types.
**Priority**: High - This would significantly improve efficiency for simple queries.

**Result**: This is the simplest guard against "walk the whole HTN even if leaf 1 solved the query."

**Implementation**:
1. First implementation: Added a `_check_success_criteria` method to the QA Specialist Agent that checks if the tool result contains the requested information. Modified the execution phase to check success criteria after each tool call and move to refinement if the criteria are met.
2. Improved implementation: Extended the plan schema with `expected_key` and `output_path` fields. Added a `_should_skip` method to the base Agent class that checks if the expected output is already in memory. Modified the planning capability to check if steps can be skipped based on success criteria. This approach works for all agents, not just the QA Specialist.

### 3. Plan-Step ↔ Tool Contract

Every plan step should declare the tool and the specific output key it expects.

| step | tool | expected key | "done" check |
|------|------|--------------|---------------|
| 1 | sec_financial_data | financial_facts.NetIncome | value not None |
| 2 | sec_financial_data (metrics) | available_metrics | list length > 0 |

**Impact**: Medium - This would make the planning more explicit about tool expectations.
**Difficulty**: Medium-High - Requires modifying the planning system to include tool expectations.
**Priority**: Medium - This is a more structural change that would improve clarity and efficiency.

**Result**: Execution skips a step automatically once the expected key is already in memory.

### 4. Merge Micro-Steps with a "Batch" Adapter

```python
def fetch_fact(metric, ticker, year):
    data = sec_financial_data(
        query_type="financial_facts",
        parameters={
            "ticker": ticker,
            "metrics": [metric],
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31"
        }
    )
    return data[metric][year]
```

**Impact**: High - This would reduce the number of steps needed for simple queries.
**Difficulty**: Medium - Requires creating adapter functions and modifying the planning system.
**Priority**: Medium-High - This would be particularly effective for factoid queries.

**Result**: The planner can fall back to HTN only when batch_capable == False.

### 5. Refinement Tokens vs. Execution Tokens

Right now the QA agent has 5 execution vs 2 refinement iterations.
For factoid Q&A you usually want the inverse:

- execution ≤ 2 (fetch, optional follow-up)
- refinement ≥ 3 (cross-check, draft, critique)

**Impact**: High - This would allow more iterations for refining the answer.
**Difficulty**: Low - Simply requires changing configuration parameters.
**Priority**: High - This is a quick change that could significantly improve answer quality.

**Result**: This removes the "out of gas before polishing" failure class.

### 6. Tool-Error Escalation Policy

get_available_metrics raised an AttributeError and the agent kept trucking.

**Quick win** → convert unknown attribute errors into an explicit "unsupported-query-type" status that bumps to refinement immediately.

**Long term** → maintain a service registry in memory: if query_type == metrics is unsupported, mark it offline so future plans skip that branch entirely.

**Impact**: Medium - This would improve error handling and prevent wasted iterations.
**Difficulty**: Medium - Requires modifying the tool execution framework to handle errors more gracefully.
**Priority**: Medium - This would improve robustness but isn't the primary issue in the current case.

### 7. Instrumentation: Cost & Latency Budget per Phase

```python
state.tokens_used['planning'|'execution'|'refinement']
state.seconds_elapsed['…']
```

**Impact**: Medium - This would provide better insights for tuning parameters.
**Difficulty**: Medium - Requires adding instrumentation to the agent framework.
**Priority**: Low - This is more of a monitoring improvement than a direct fix.

**Result**: You'll quickly see whether planning or refinement needs bigger budgets, and you can auto-scale max_*_iterations based on token ceilings rather than fixed integers.