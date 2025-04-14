# Visualization Scripts

This directory contains scripts for visualizing SEC filing data and workflow logs.

## Available Scripts

### Database Visualization

- `launch_duckdb_explorer.py`: Launch DuckDB Explorer
- `launch_duckdb_web.py`: Launch DuckDB Web Explorer
- `simple_duckdb_explorer.py`: Simple DuckDB Explorer
- `streamlit_duckdb_explorer.py`: Streamlit DuckDB Explorer

### Data Store Visualization

- `explore_vector_store.py`: Explore vector store
- `explore_neo4j_graph.py`: Explore Neo4j graph database

### Log Visualization

- `workflow_visualizer.py`: Streamlit app for visualizing logs
- `launch_log_visualizer.py`: Launch the Log Visualizer

## Log Visualization

The SEC Filing Analyzer Log Visualizer provides interactive visualizations of agent and workflow logs, including:

- Timeline of workflow steps
- Timing analysis of operations
- Tool call analysis
- Agent interactions
- LLM interactions (prompts and responses with token usage)
- Raw log entries

### Features

#### Workflow Steps Visualization
Displays a timeline of workflow steps, showing when each step started and how long it took to complete. This helps identify bottlenecks in the workflow process.

#### Timing Analysis
Provides detailed timing information for various operations, including tool execution times, LLM generation times, and step execution times. The visualizer shows both total duration by category and a timeline of individual operations.

#### Tool Calls Analysis
Shows all tool calls made during the workflow, including the tool name, arguments, and execution time. This helps understand which tools are used most frequently and how they contribute to the overall workflow.

#### LLM Interactions
Displays all LLM interactions during the workflow, including:
- Prompts sent to the LLM
- Responses received from the LLM
- Token usage statistics (prompt tokens, completion tokens, total tokens)

This section is particularly useful for understanding the reasoning process of the agents and identifying opportunities for prompt optimization.

### Usage

```bash
# Launch the Log Visualizer
python scripts/visualization/launch_log_visualizer.py

# Visualize a specific log file
python scripts/visualization/launch_log_visualizer.py --log-file data/logs/agents/FinancialAnalystAgent_20250414_125810.log

# Visualize a workflow log file
python scripts/visualization/launch_log_visualizer.py --log-file data/logs/workflows/workflow_FullWorkflowDemo_20250414_132500.log
```

### Supported Log Types

The visualizer supports both agent logs and workflow logs:

- **Agent Logs**: Logs from individual agents (QA Specialist, Financial Analyst, Risk Analyst)
- **Workflow Logs**: Logs from workflow runs that coordinate multiple agents

The visualizer automatically detects the log type and displays the appropriate visualizations.

### Generating Sample Workflow Logs

For testing purposes, you can generate sample workflow logs:

```bash
# Generate a sample workflow log
python scripts/tests/generate_sample_workflow_log.py

# Specify a custom log directory
python scripts/tests/generate_sample_workflow_log.py --log-dir logs/workflows
```
