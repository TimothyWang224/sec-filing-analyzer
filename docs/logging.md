# Logging and Performance Monitoring

The SEC Filing Analyzer uses a standardized logging system to track operations, errors, agent activities, and performance metrics.

## Log Directory Structure

All logs are stored in the `data/logs/` directory with the following subdirectory structure:

- `data/logs/agents/`: Logs from agent activities (QA Specialist, Financial Analyst, Risk Analyst, Coordinator)
- `data/logs/tests/`: Logs from test runs
- `data/logs/general/`: General application logs

## Log Formats

Logs are stored in two formats:

1. **Plain Text (`.log`)**: Human-readable log files with timestamps, log levels, and messages
2. **Structured JSON (`.json`)**: Machine-readable log files with structured data for programmatic analysis

## Agent Logging

Agent logs include:

- Agent initialization and configuration
- Tool usage and results
- Memory updates
- Reasoning steps
- Final outputs

### Log File Naming

Agent log files follow this naming convention:

```
{AgentType}_{YYYYMMDD}_{HHMMSS}.log
{AgentType}_{YYYYMMDD}_{HHMMSS}.json
```

For example:
- `QASpecialistAgent_20250414_104954.log`
- `FinancialAnalystAgent_20250414_105023.json`

## Configuring Logging

You can configure logging when initializing agents:

```python
from src.capabilities.logging import LoggingCapability
from src.sec_filing_analyzer.utils.logging_utils import get_standard_log_dir

# Get the standard log directory for agents
log_dir = str(get_standard_log_dir("agents"))

# Create a logging capability with custom settings
logging_capability = LoggingCapability(
    log_dir=log_dir,
    log_level="DEBUG",  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_to_console=True,
    log_to_file=True,
    include_memory=True,
    include_context=True,
    include_actions=True,
    include_results=True,
    include_prompts=False,  # Set to True to include LLM prompts (may contain sensitive data)
    include_responses=False,  # Set to True to include full LLM responses
    max_content_length=1000  # Limit content length to avoid huge logs
)
```

## Analyzing Logs

### Plain Text Logs

Plain text logs can be analyzed using standard text processing tools:

```bash
# View the most recent log file
cat $(ls -t data/logs/agents/*.log | head -1)

# Search for errors in all log files
grep "ERROR" data/logs/agents/*.log

# Count occurrences of specific events
grep -c "Tool called" data/logs/agents/*.log
```

### JSON Logs

JSON logs can be analyzed using tools like `jq` or programmatically:

```bash
# Extract all tool calls from a JSON log
jq '.logs[] | select(.type == "tool_call")' data/logs/agents/QASpecialistAgent_20250414_104954.json

# Find all errors
jq '.logs[] | select(.level == "ERROR")' data/logs/agents/*.json
```

## Performance Monitoring

The system includes built-in performance monitoring through detailed timing logs. These logs track the execution time of various operations, helping identify bottlenecks and performance issues.

### Timing Information

Timing information is logged for key operations:

- **LLM Calls**: Time spent generating responses from language models
- **Tool Execution**: Time spent executing individual tools
- **Processing Steps**: Time spent in different phases of processing

Each timing log entry includes:

- Category (e.g., "llm", "tool", "process")
- Operation name
- Duration in seconds

Example timing log entries:

```
2025-04-14 10:50:12,345 - agent.QASpecialistAgent.20250414_105012 - INFO - TIMING: llm:generate completed in 2.345s
2025-04-14 10:50:15,678 - agent.QASpecialistAgent.20250414_105012 - INFO - TIMING: tool:sec_semantic_search completed in 1.234s
```

### Analyzing Performance

A utility script is provided to analyze timing information in logs:

```bash
# Analyze the latest log file
python scripts/utils/analyze_timing.py --latest

# Analyze a specific log file
python scripts/utils/analyze_timing.py --log-file data/logs/agents/QASpecialistAgent_20250414_105012.log

# Generate visualizations
python scripts/utils/analyze_timing.py --latest --output-dir data/analysis
```

The script provides:

1. **Timing Summary**: Breakdown of time spent in different categories and operations
2. **Bottleneck Identification**: Highlights operations that may be performance bottlenecks
3. **Visualizations**: Charts showing time distribution and operation timeline
4. **Recommendations**: Suggestions for improving performance

## Log Rotation and Maintenance

The system does not currently implement automatic log rotation. For production environments, consider implementing log rotation using tools like `logrotate` or a custom script to prevent logs from consuming too much disk space.

A simple maintenance script could:

1. Compress logs older than a certain date
2. Delete logs older than a retention period
3. Move logs to an archival storage system

## Troubleshooting

If you're not seeing logs where expected:

1. Check that the log directory exists and is writable
2. Verify the log level is appropriate (DEBUG shows more information than INFO)
3. Ensure the logging capability is properly initialized and added to the agent
4. Check that `log_to_file` is set to `True` in the LoggingCapability configuration
