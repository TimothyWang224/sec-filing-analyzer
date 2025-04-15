# Agent Configuration Integration

This document describes the integration of agent parameters with the master configuration system.

## Overview

Previously, agent parameters were hardcoded in each agent's initialization method. This approach had several drawbacks:
- Parameters were scattered across multiple files
- Changes required updating code in multiple places
- There was no centralized way to manage agent parameters

The new approach integrates all agent parameters with the master configuration system, providing a centralized way to manage agent parameters.

## Changes Made

### 1. Master Config Files

The following master config files were updated:

#### `src/sec_filing_analyzer/llm/llm_config.py`
- Added a `BASE_AGENT_CONFIG` dictionary with default values for all agent parameters
- Updated all agent configurations to include phase-based parameters
- Added tool execution parameters to all agent configurations
- Updated model names to reflect the latest available models (gpt-4o, gpt-4o-mini)
- Enhanced the `get_recommended_config` method to adjust all relevant parameters based on task complexity

#### `src/sec_filing_analyzer/config.py`
- Added a new `AgentConfig` class to store agent-related configuration
- Added environment variable support for all agent parameters
- Added a `from_env` method to load agent configuration from environment variables
- Added an `AGENT_CONFIG` dictionary for backward compatibility

#### `data/config/etl_config.json`
- Added a new "agent" section with all agent parameters

### 2. Agent Files

The following agent files were updated to use the master config:

#### `src/agents/coordinator.py`
- Added imports for the config modules
- Added a try-except block to handle potential import errors
- Updated the specialized agent initialization to use configuration values
- Added fallback to environment variables and default values

#### `src/agents/financial_analyst.py`, `src/agents/risk_analyst.py`, `src/agents/qa_specialist.py`
- Added imports for the config modules
- Added a try-except block to handle potential import errors
- Updated the agent initialization to use configuration values
- Added fallback to environment variables and default values

## Configuration Hierarchy

The configuration values are resolved in the following order:
1. Values passed directly to the agent constructor
2. Values from the agent-specific configuration in `llm_config.py`
3. Values from the global `AGENT_CONFIG` in `config.py`
4. Values from environment variables
5. Default hardcoded values

This hierarchy allows for flexible configuration at different levels.

## Environment Variables

The following environment variables can be used to configure agent parameters:

### Global Agent Parameters
- `AGENT_MAX_ITERATIONS`: Maximum number of iterations (legacy parameter)
- `AGENT_MAX_PLANNING_ITERATIONS`: Maximum number of planning iterations
- `AGENT_MAX_EXECUTION_ITERATIONS`: Maximum number of execution iterations
- `AGENT_MAX_REFINEMENT_ITERATIONS`: Maximum number of refinement iterations
- `AGENT_MAX_TOOL_RETRIES`: Maximum number of tool retries
- `AGENT_TOOLS_PER_ITERATION`: Number of tools to use per iteration
- `AGENT_CIRCUIT_BREAKER_THRESHOLD`: Circuit breaker threshold
- `AGENT_CIRCUIT_BREAKER_RESET_TIMEOUT`: Circuit breaker reset timeout
- `AGENT_MAX_DURATION_SECONDS`: Maximum duration in seconds
- `AGENT_ENABLE_DYNAMIC_TERMINATION`: Enable dynamic termination
- `AGENT_MIN_CONFIDENCE_THRESHOLD`: Minimum confidence threshold
- `DEFAULT_LLM_MODEL`: Default LLM model
- `DEFAULT_LLM_TEMPERATURE`: Default LLM temperature
- `DEFAULT_LLM_MAX_TOKENS`: Default LLM max tokens

### Agent-Specific Parameters
Each agent type has its own set of environment variables with the same pattern:

#### Financial Analyst
- `FINANCIAL_ANALYST_MAX_ITERATIONS`
- `FINANCIAL_ANALYST_PLANNING_ITERATIONS`
- `FINANCIAL_ANALYST_EXECUTION_ITERATIONS`
- `FINANCIAL_ANALYST_REFINEMENT_ITERATIONS`
- `FINANCIAL_ANALYST_MAX_TOOL_RETRIES`
- `FINANCIAL_ANALYST_TOOLS_PER_ITERATION`
- `FINANCIAL_ANALYST_MAX_DURATION_SECONDS`
- `FINANCIAL_ANALYST_MODEL`
- `FINANCIAL_ANALYST_TEMPERATURE`
- `FINANCIAL_ANALYST_MAX_TOKENS`
- `FINANCIAL_ANALYST_ENABLE_DYNAMIC_TERMINATION`
- `FINANCIAL_ANALYST_MIN_CONFIDENCE_THRESHOLD`

#### Risk Analyst
- `RISK_ANALYST_MAX_ITERATIONS`
- `RISK_ANALYST_PLANNING_ITERATIONS`
- `RISK_ANALYST_EXECUTION_ITERATIONS`
- `RISK_ANALYST_REFINEMENT_ITERATIONS`
- `RISK_ANALYST_MAX_TOOL_RETRIES`
- `RISK_ANALYST_TOOLS_PER_ITERATION`
- `RISK_ANALYST_MAX_DURATION_SECONDS`
- `RISK_ANALYST_MODEL`
- `RISK_ANALYST_TEMPERATURE`
- `RISK_ANALYST_MAX_TOKENS`
- `RISK_ANALYST_ENABLE_DYNAMIC_TERMINATION`
- `RISK_ANALYST_MIN_CONFIDENCE_THRESHOLD`

#### QA Specialist
- `QA_SPECIALIST_MAX_ITERATIONS`
- `QA_SPECIALIST_PLANNING_ITERATIONS`
- `QA_SPECIALIST_EXECUTION_ITERATIONS`
- `QA_SPECIALIST_REFINEMENT_ITERATIONS`
- `QA_SPECIALIST_MAX_TOOL_RETRIES`
- `QA_SPECIALIST_TOOLS_PER_ITERATION`
- `QA_SPECIALIST_MAX_DURATION_SECONDS`
- `QA_SPECIALIST_MODEL`
- `QA_SPECIALIST_TEMPERATURE`
- `QA_SPECIALIST_MAX_TOKENS`
- `QA_SPECIALIST_ENABLE_DYNAMIC_TERMINATION`
- `QA_SPECIALIST_MIN_CONFIDENCE_THRESHOLD`

## Error Handling

The configuration system includes robust error handling:
- If the config imports fail, the code falls back to environment variables
- If environment variables aren't set, it uses hardcoded default values
- Each level of the configuration hierarchy provides a fallback to the next level

## IDE Warnings

The changes introduced some IDE warnings that should be addressed:

### Unused Imports
- `import re` in multiple agent files
- `import os` (when not using environment variables)

### Unused Variables
- Variables declared but not used in various agent methods

These warnings don't affect functionality but should be cleaned up for code quality.

## Future Improvements

1. **Configuration Validation**: Add validation logic to ensure that configuration values are within acceptable ranges.
2. **Configuration Documentation**: Create a comprehensive documentation file that explains all configuration parameters and their effects.
3. **Configuration UI**: Consider creating a simple UI for editing configuration files to make it easier for users to customize the system.
4. **Configuration Versioning**: Add version information to configuration files to track changes over time.
5. **Default Configuration Files**: Create default configuration files that users can copy and modify.
