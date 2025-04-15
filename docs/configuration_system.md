# Configuration System

This document describes the configuration system for the SEC Filing Analyzer project.

## Overview

The SEC Filing Analyzer uses a unified configuration system that provides a centralized way to manage configuration values across the application. The configuration system is designed to be flexible, allowing configuration values to be specified in multiple ways:

1. **Configuration Files**: Configuration values can be specified in JSON files.
2. **Environment Variables**: Configuration values can be specified using environment variables.
3. **Direct Parameters**: Configuration values can be passed directly to functions and classes.

The configuration system uses a hierarchical approach, with more specific configurations overriding more general ones.

## Configuration Provider

The `ConfigProvider` class is the central component of the configuration system. It provides a unified interface for accessing configuration values from various sources.

### Usage

```python
from sec_filing_analyzer.config import ConfigProvider

# Initialize the ConfigProvider
ConfigProvider.initialize()

# Get configuration for a specific agent type
config = ConfigProvider.get_agent_config("financial_analyst")

# Get a specific configuration value
model = config.get("model", "default_model")
```

### Configuration Hierarchy

The configuration values are resolved in the following order:

1. Values passed directly to functions and classes
2. Values from agent-specific configurations in `llm_config.py`
3. Values from the global `AGENT_CONFIG` in `config.py`
4. Values from external configuration files
5. Values from environment variables
6. Default hardcoded values

This hierarchy allows for flexible configuration at different levels.

## Agent Configuration

Agent configuration is a key part of the configuration system. Each agent type has its own configuration, which includes parameters for:

- Agent iteration parameters
- Tool execution parameters
- Runtime parameters
- LLM parameters
- Termination parameters

### Agent Types

The following agent types are available:

- `coordinator`: Coordinates multiple agents for comprehensive financial diligence
- `financial_analyst`: Analyzes financial statements and metrics
- `risk_analyst`: Identifies and analyzes financial and operational risks
- `qa_specialist`: Answers financial questions and provides detailed explanations
- `sec_analysis`: Analyzes SEC filings

### Agent Configuration Parameters

Each agent configuration includes the following parameters:

#### Agent Iteration Parameters
- `max_iterations`: Maximum number of iterations (legacy parameter)
- `max_planning_iterations`: Maximum number of planning iterations
- `max_execution_iterations`: Maximum number of execution iterations
- `max_refinement_iterations`: Maximum number of refinement iterations

#### Tool Execution Parameters
- `max_tool_retries`: Maximum number of tool retries
- `tools_per_iteration`: Number of tools to use per iteration
- `circuit_breaker_threshold`: Circuit breaker threshold
- `circuit_breaker_reset_timeout`: Circuit breaker reset timeout

#### Runtime Parameters
- `max_duration_seconds`: Maximum duration in seconds

#### LLM Parameters
- `model`: LLM model to use
- `temperature`: Temperature for LLM generation
- `max_tokens`: Maximum tokens for LLM generation
- `system_prompt`: System prompt for LLM

#### Termination Parameters
- `enable_dynamic_termination`: Whether to enable dynamic termination
- `min_confidence_threshold`: Minimum confidence threshold for satisfactory results

### Environment Variables

The following environment variables can be used to configure agent parameters:

#### Global Agent Parameters
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

#### Agent-Specific Parameters
Each agent type has its own set of environment variables with the same pattern:

##### Financial Analyst
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

##### Risk Analyst
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

##### QA Specialist
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

##### Coordinator
- `COORDINATOR_MAX_ITERATIONS`
- `COORDINATOR_PLANNING_ITERATIONS`
- `COORDINATOR_EXECUTION_ITERATIONS`
- `COORDINATOR_REFINEMENT_ITERATIONS`
- `COORDINATOR_MAX_TOOL_RETRIES`
- `COORDINATOR_TOOLS_PER_ITERATION`
- `COORDINATOR_MAX_DURATION_SECONDS`
- `COORDINATOR_MODEL`
- `COORDINATOR_TEMPERATURE`
- `COORDINATOR_MAX_TOKENS`
- `COORDINATOR_ENABLE_DYNAMIC_TERMINATION`
- `COORDINATOR_MIN_CONFIDENCE_THRESHOLD`

## External Configuration Files

External configuration files can be used to specify configuration values. The default location for external configuration files is `data/config/etl_config.json`.

### Example Configuration File

```json
{
  "etl_pipeline": {
    "process_semantic": true,
    "process_quantitative": true,
    "use_parallel": true,
    "max_workers": 4,
    "batch_size": 50,
    "rate_limit": 0.2,
    "max_retries": 3,
    "delay_between_companies": 1,
    "chunk_size": 1024,
    "chunk_overlap": 50,
    "embedding_model": "text-embedding-3-small"
  },
  "agent": {
    "max_iterations": 3,
    "max_planning_iterations": 2,
    "max_execution_iterations": 3,
    "max_refinement_iterations": 1,
    "max_tool_retries": 2,
    "tools_per_iteration": 1,
    "circuit_breaker_threshold": 3,
    "circuit_breaker_reset_timeout": 300,
    "max_duration_seconds": 180,
    "enable_dynamic_termination": false,
    "min_confidence_threshold": 0.8,
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0.7,
    "llm_max_tokens": 4000
  }
}
```

## LLM Configuration Factory

The `LLMConfigFactory` class provides a factory for creating and managing LLM configurations. It includes methods for:

- Creating configurations for specific agent types
- Creating configurations with different task complexities
- Validating configurations

### Usage

```python
from sec_filing_analyzer.llm.llm_config import LLMConfigFactory

# Create a configuration for a specific agent type
config = LLMConfigFactory.create_config("financial_analyst")

# Create a configuration with a specific task complexity
config = LLMConfigFactory.get_recommended_config("coordinator", task_complexity="high")

# Validate a configuration
is_valid = LLMConfigFactory.validate_config(config)
```

## Task Complexity

The configuration system supports different task complexities, which can be used to adjust the configuration based on the complexity of the task:

- `low`: For simple tasks
- `medium`: For moderate tasks
- `high`: For complex tasks

Each task complexity level has different configuration values for:

- LLM model
- Maximum tokens
- Maximum planning iterations
- Maximum execution iterations
- Maximum refinement iterations
- Maximum duration in seconds

### Example

```python
from sec_filing_analyzer.llm.llm_config import LLMConfigFactory

# Get a configuration for a high-complexity task
config = LLMConfigFactory.get_recommended_config("coordinator", task_complexity="high")
```

## Error Handling

The configuration system includes robust error handling:

- If the config imports fail, the code falls back to environment variables
- If environment variables aren't set, it uses hardcoded default values
- Each level of the configuration hierarchy provides a fallback to the next level

## Future Improvements

1. **Configuration Validation**: Add validation logic to ensure that configuration values are within acceptable ranges.
2. **Configuration Documentation**: Create a comprehensive documentation file that explains all configuration parameters and their effects.
3. **Configuration UI**: Consider creating a simple UI for editing configuration files to make it easier for users to customize the system.
4. **Configuration Versioning**: Add version information to configuration files to track changes over time.
5. **Default Configuration Files**: Create default configuration files that users can copy and modify.
