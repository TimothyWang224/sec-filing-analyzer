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

### ✅ 3. Plan-Step ↔ Tool Contract

Every plan step should declare the tool and the specific output key it expects.

| step | tool | expected key | "done" check |
|------|------|--------------|---------------|
| 1 | sec_financial_data | financial_facts.NetIncome | value not None |
| 2 | sec_financial_data (metrics) | available_metrics | list length > 0 |

**Impact**: Medium - This would make the planning more explicit about tool expectations.
**Difficulty**: Medium-High - Requires modifying the planning system to include tool expectations.
**Priority**: Medium - This is a more structural change that would improve clarity and efficiency.

**Result**: Execution skips a step automatically once the expected key is already in memory.

**Implementation**: Created a formal contract between plan steps and tools using Pydantic models. Added `ToolSpec` and `PlanStep` models in `contracts.py` to define the interface. Updated the `ToolRegistry` to use the `ToolSpec` model for tool specifications. Modified the `_execute_current_step` method in the base Agent class to use the tool specifications and standardized memory access patterns. Added the `extract_value` function to standardize how tools store and retrieve data from memory.

**Enhanced Implementation**: Added standardized parameter models for all tools using Pydantic. Implemented a comprehensive error hierarchy with user-friendly error messages. Created a validation framework to validate tool calls before execution. Updated all tools to use the new parameter models and validation framework. Added success short-circuit to skip steps when the expected output is already in memory.

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

### ✅ 5. Refinement Tokens vs. Execution Tokens

Right now the QA agent has 5 execution vs 2 refinement iterations.
For factoid Q&A you usually want the inverse:

- execution ≤ 2 (fetch, optional follow-up)
- refinement ≥ 3 (cross-check, draft, critique)

**Impact**: High - This would allow more iterations for refining the answer.
**Difficulty**: Low - Simply requires changing configuration parameters.
**Priority**: High - This is a quick change that could significantly improve answer quality.

**Result**: This removes the "out of gas before polishing" failure class.

**Implementation**: Implemented a token-based budgeting system that allocates tokens to each phase (planning, execution, refinement) with a 10/40/50 split by default. Added the ability to roll over unused tokens from execution to refinement, ensuring that simple queries that need minimal execution can benefit from more thorough refinement. Replaced iteration-based control with token-based control, while keeping iteration limits as a secondary safety mechanism.

### 6. Tool-Error Escalation Policy

get_available_metrics raised an AttributeError and the agent kept trucking.

**Quick win** → convert unknown attribute errors into an explicit "unsupported-query-type" status that bumps to refinement immediately.

**Long term** → maintain a service registry in memory: if query_type == metrics is unsupported, mark it offline so future plans skip that branch entirely.

**Impact**: Medium - This would improve error handling and prevent wasted iterations.
**Difficulty**: Medium - Requires modifying the tool execution framework to handle errors more gracefully.
**Priority**: Medium - This would improve robustness but isn't the primary issue in the current case.

### ⚠️ 7. Instrumentation: Cost & Latency Budget per Phase

```python
state.tokens_used['planning'|'execution'|'refinement']
state.seconds_elapsed['…']
```

**Impact**: Medium - This would provide better insights for tuning parameters.
**Difficulty**: Medium - Requires adding instrumentation to the agent framework.
**Priority**: Low - This is more of a monitoring improvement than a direct fix.

**Result**: You'll quickly see whether planning or refinement needs bigger budgets, and you can auto-scale max_*_iterations based on token ceilings rather than fixed integers.

**Partial Implementation**: Implemented token tracking with `state.tokens_used` for each phase. Still need to implement time tracking with `state.seconds_elapsed`.

## Stability Fixes

### 1. Fix JSON Serialization in LoggingCapability ✅

**Issue**: `Object of type Plan is not JSON serializable` error in LoggingCapability

**Fix**:
- ✅ Added `_prepare_for_serialization` method to convert Pydantic objects to dictionaries
- ✅ Added `default=str` to `json.dump()` calls to handle non-serializable objects

**Impact**: High - This prevents errors during logging that can disrupt the agent's execution flow
**Difficulty**: Low - Simple code change
**Priority**: High - This is causing the infinite planning loop

### 2. Prevent Infinite Planning Loop ✅

**Issue**: Agent stuck in planning phase, never switches to execution

**Fix**:
- ✅ Added a guard to check if plan status is "in_progress" and no changes, then jump to execution
- ✅ Added code to roll over unused tokens from planning to execution

**Impact**: High - Prevents the agent from getting stuck in planning
**Difficulty**: Low - Simple conditional check
**Priority**: High - Critical for agent functionality

### 3. Fix Vector Store Initialization ✅

**Issue**: Vector store initializes with 0 documents/companies

**Fix**:
- ✅ Added explicit check for metadata.json file
- ✅ Added clear error message if metadata.json doesn't exist
- ✅ Added instructions to run ETL pipeline or check VECTOR_STORE_DIR path

**Impact**: High - Required for semantic search functionality
**Difficulty**: Medium - May require path handling changes
**Priority**: Medium-High - Needed for complete functionality

### 4. Fix Duplicate Schema Registry Mappings ✅

**Issue**: Schema registry mappings registered multiple times

**Fix**:
- ✅ Added a check to skip if mapping already exists
- ✅ Added warning if trying to overwrite with a different mapping

**Impact**: Low - Just reduces log noise
**Difficulty**: Low - Simple guard clause
**Priority**: Low - Not affecting functionality

### 5. Fix Module Import Warning ✅

**Issue**: `sec_filing_analyzer.tools module not found, using src.tools instead`

**Fix**:
- ✅ Added a re-export file in sec_filing_analyzer/tools/__init__.py

**Impact**: Low - Just reduces log noise
**Difficulty**: Low - Simple file addition or import changes
**Priority**: Low - Not affecting functionality

### 6. Set Default Token Budgets ✅

**Issue**: Token budgets configured as None

**Fix**:
- ✅ Added DEFAULT_TOKEN_BUDGET constant to AgentState class
- ✅ Updated _configure_token_budgets method to use the default budget
- ✅ Initialized token_budget with DEFAULT_TOKEN_BUDGET in AgentState constructor
- ✅ Increased token budget to 250,000 tokens (25k planning, 100k execution, 125k refinement)
- ✅ Created global config.json file with token budget settings
- ✅ Updated ConfigProvider to prioritize config.json and properly load token_budgets

**Impact**: Medium - Ensures token budgeting works correctly
**Difficulty**: Low - Simple default value addition
**Priority**: Medium - Affects optimization but not core functionality

### 7. Improve Plan Step Keys and Paths ✅

**Issue**: Generic expected_key and incorrect output_path

**Fix**:
- ✅ Updated _validate_and_clean_plan to generate more specific keys like `"{ticker}_{metric}_{year}"`
- ✅ Enhanced output_path generation to match the actual structure returned by tools
- ✅ Added special handling for different query types and parameters

**Impact**: High - Critical for success criteria checking
**Difficulty**: Medium - Requires changes to planning logic
**Priority**: High - Affects core functionality

### 8. Reduce Log Noise ✅

**Issue**: Faiss GPU/AVX512 warnings and Neo4j constraint warnings

**Fix**:
- ✅ Added configure_noisy_loggers function to set logger level to WARNING for noisy libraries
- ✅ Called configure_noisy_loggers from setup_logging

**Impact**: Low - Just improves log readability
**Difficulty**: Low - Simple logger configuration
**Priority**: Low - Not affecting functionality

### 9. Remove Redundant Validation Steps in Planning ✅

**Issue**: Agent creates redundant validation steps that re-check data already retrieved

**Fix**:
- ✅ Updated planning prompt to explicitly discourage redundant validation steps
- ✅ Added clear warning in the prompt about automatic validation
- ✅ Updated system prompt for plan generation to reinforce the message
- ✅ Updated reflection prompt to discourage adding validation steps
- ✅ Improved default plan to be more specific and efficient

**Impact**: Medium - Improves efficiency and reduces unnecessary steps
**Difficulty**: Low - Prompt modifications only
**Priority**: Medium - Improves user experience

### 10. Fix PlanStep Dependencies Validation Error ✅

**Issue**: Error: `1 validation error for PlanStep dependencies.0 Input should be a valid integer`

**Fix**:
- ✅ Updated `_dict_to_plan_step` method to properly handle dependencies
- ✅ Enhanced `_validate_and_clean_plan` to ensure dependencies are always integers
- ✅ Added robust error handling to prevent crashes when invalid dependencies are encountered

**Impact**: High - Prevents crashes during plan creation and execution
**Difficulty**: Low - Simple validation and conversion logic
**Priority**: High - Critical for agent functionality

### 11. Fix Coordinator Max Refinement Iterations Issue ✅

**Issue**: Financial Diligence Coordinator immediately terminates when entering refinement phase due to max_refinement_iterations

**Fix**:
- ✅ Modified `set_phase` method in AgentState to reset phase iteration counter when changing phases
- ✅ Added check to only reset counter if actually changing to a different phase
- ✅ Fixed `should_terminate` method to check phase-specific iteration counters instead of global iteration counter
- ✅ Added more detailed logging with current/max iteration counts for better debugging
- ✅ Ensures each phase starts with a clean iteration count and is properly tracked

**Impact**: High - Allows coordinator to properly perform refinement iterations
**Difficulty**: Low - Simple modification to phase transition logic
**Priority**: High - Critical for proper agent functionality

### 12. Remove Mock Data from Financial Data Tool ✅

**Issue**: SEC Financial Data Tool was returning hardcoded mock data with incorrect values (e.g., GOOGL revenue reported as $383.29B instead of $307.39B)

**Fix**:
- ✅ Removed all hardcoded mock data from the SEC Financial Data Tool
- ✅ Added proper error handling for database connection failures
- ✅ Added warning messages for cases where no data is found
- ✅ Ensured consistent error reporting across all query methods
- ✅ Tool now returns empty results with error messages instead of fake data

**Impact**: High - Prevents misleading information in financial reports
**Difficulty**: Medium - Required updating multiple methods with consistent error handling
**Priority**: High - Critical for data accuracy in production

### 13. Fix Database Path Inconsistency in Financial Data Tool ✅

**Issue**: SEC Financial Data Tool was using a hardcoded database path that didn't match the path used by other components

**Fix**:
- ✅ Updated SECFinancialDataTool to use the ETLConfig from ConfigProvider
- ✅ Added more detailed logging for database connection failures
- ✅ Removed misleading log message about fallback mode with mock data
- ✅ Ensured consistent database path usage across the application
- ✅ Linked database path to the master configuration system

**Impact**: High - Enables the tool to find and use the correct database
**Difficulty**: Low - Simple modification to use the centralized configuration
**Priority**: High - Critical for accessing financial data

### 14. Fix Missing Database File Issue ✅

**Issue**: Configuration was pointing to a non-existent database file (`improved_financial_data.duckdb`) that was purged during GitHub upload

**Fix**:
- ✅ Located database schema files that survived the purge
- ✅ Fixed SQL syntax errors in the schema file
- ✅ Updated the database initialization script
- ✅ Successfully recreated the `improved_financial_data.duckdb` database with the correct schema
- ✅ Verified the database schema using the check_db_schema.py script

**Impact**: High - Enables the system to use the correct database structure
**Difficulty**: Medium - Required fixing schema files and initialization scripts
**Priority**: High - Critical for system functionality

### 15. Optimize Vector Store for GPU Acceleration ✅

**Issue**: Embeddings were stored as JSON files instead of NumPy binaries, resulting in slower processing during semantic search

**Fix**:
- ✅ Created a script to migrate existing JSON embeddings to NumPy binary format
- ✅ Successfully migrated 30,483 embedding files to NumPy format
- ✅ Updated ETL pipeline to use OptimizedVectorStore by default
- ✅ Enabled GPU acceleration for FAISS index
- ✅ Rebuilt the FAISS index using the NumPy binary files
- ✅ Updated configuration to use GPU acceleration by default

**Impact**: High - Significantly improves semantic search performance
**Difficulty**: Medium - Required creating migration scripts and updating configuration
**Priority**: Medium - Enhances system performance