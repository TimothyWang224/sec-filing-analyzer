"""
Workflow Visualizer

A Streamlit app for visualizing workflow logs and intermediate steps.
This tool parses workflow log files and generates interactive visualizations
of the workflow steps, timing, and results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Set page config
st.set_page_config(
    page_title="SEC Filing Analyzer Log Visualizer",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("SEC Filing Analyzer Log Visualizer")

# Sidebar
st.sidebar.header("Configuration")

# Log file selection
def get_log_files():
    """Get available log files."""
    log_files = []

    # Check for workflow logs
    workflow_log_dir = Path("data/logs/workflows")
    if workflow_log_dir.exists():
        log_files.extend([
            f for f in workflow_log_dir.glob("workflow_*.log")
        ])

    # Check for agent logs
    agent_log_dir = Path("data/logs/agents")
    if agent_log_dir.exists():
        log_files.extend([
            f for f in agent_log_dir.glob("*.log")
        ])

    # Check alternate locations
    alt_log_dir = Path("logs")
    if alt_log_dir.exists():
        for subdir in ["workflows", "agents"]:
            subdir_path = alt_log_dir / subdir
            if subdir_path.exists():
                log_files.extend([
                    f for f in subdir_path.glob("*.log")
                ])

    # Sort by modification time (newest first)
    return sorted(
        log_files,
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

log_files = get_log_files()

# Create more user-friendly display names for the log files
log_file_display = {}
for f in log_files:
    # Extract the file name without extension
    file_name = f.stem

    # Determine the log type
    if "workflow" in file_name.lower():
        log_type = "Workflow"
    else:
        log_type = "Agent"

    # Extract the timestamp if available
    timestamp_match = re.search(r"(\d{8}_\d{6})", file_name)
    if timestamp_match:
        timestamp_str = timestamp_match.group(1)
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            formatted_time = timestamp_str
    else:
        formatted_time = "Unknown"

    # Create a display name
    display_name = f"{log_type}: {file_name} ({formatted_time})"
    log_file_display[str(f)] = display_name

log_file_options = [str(f) for f in log_files]

if not log_file_options:
    st.error("No log files found. Please check the logs directory.")
    st.stop()

selected_log_file = st.sidebar.selectbox(
    "Select Log File",
    options=log_file_options,
    format_func=lambda x: log_file_display[x],
    index=0
)

# Display log file info
selected_file = Path(selected_log_file)
st.sidebar.info(
    f"File: {selected_file.name}\n"
    f"Size: {selected_file.stat().st_size / 1024:.1f} KB\n"
    f"Modified: {datetime.fromtimestamp(selected_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
)

# Parse log file
@st.cache_data
def parse_log_file(log_file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse a workflow log file.

    Args:
        log_file_path: Path to the log file

    Returns:
        Tuple of (log entries, workflow metadata)
    """
    log_entries = []
    workflow_metadata = {
        "workflow_id": "",
        "start_time": None,
        "end_time": None,
        "status": "unknown",
        "agents": set()
    }

    # Regular expression patterns
    timestamp_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    log_pattern = rf"{timestamp_pattern} - ([^-]+) - ([A-Z]+) - (.+)"
    timing_pattern = r"TIMING: ([^:]+):([^ ]+) completed in ([0-9.]+)s"
    tool_pattern = r"Executing tool call \d+/\d+: ([^ ]+)"
    tool_args_pattern = r"Tool arguments: (.+)"
    step_pattern = r"Step: ([^-]+) - (.*)"
    llm_prompt_pattern = r"LLM Prompt: (.+)"
    llm_response_pattern = r"LLM Response: tokens=(\d+) \(prompt=(\d+), completion=(\d+)\)"
    llm_content_pattern = r"LLM Content: (.+)"

    # Try to find corresponding JSON log file
    json_log_path = Path(log_file_path).with_suffix(".json")
    if json_log_path.exists():
        try:
            with open(json_log_path, 'r') as f:
                json_data = json.load(f)
                workflow_metadata.update({
                    "workflow_id": json_data.get("workflow_id", ""),
                    "start_time": json_data.get("start_time"),
                    "end_time": json_data.get("end_time"),
                    "status": json_data.get("status", "unknown")
                })
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    # Parse the log file
    with open(log_file_path, 'r') as f:
        current_step = None
        current_tool = None
        current_tool_args = None
        current_llm_prompt = None
        current_llm_tokens = None
        current_llm_id = None
        llm_interactions = []

        for line in f:
            match = re.match(log_pattern, line)
            if not match:
                continue

            timestamp_str, logger_name, level, message = match.groups()
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

            # Extract workflow ID from logger name if not already set
            if not workflow_metadata["workflow_id"] and "." in logger_name:
                workflow_metadata["workflow_id"] = logger_name.split(".", 1)[1]

            # Track agents
            if "agent." in logger_name:
                agent_name = logger_name.split(".", 1)[1]
                workflow_metadata["agents"].add(agent_name)

            # Parse timing information
            timing_match = re.search(timing_pattern, message)
            if timing_match:
                category, operation, duration = timing_match.groups()
                log_entries.append({
                    "timestamp": timestamp,
                    "type": "timing",
                    "category": category,
                    "operation": operation,
                    "duration": float(duration),
                    "message": message,
                    "level": level,
                    "logger": logger_name
                })
                continue

            # Parse tool execution
            tool_match = re.search(tool_pattern, message)
            if tool_match:
                current_tool = tool_match.group(1)
                continue

            tool_args_match = re.search(tool_args_pattern, message)
            if tool_args_match and current_tool:
                current_tool_args = tool_args_match.group(1)
                log_entries.append({
                    "timestamp": timestamp,
                    "type": "tool",
                    "tool": current_tool,
                    "args": current_tool_args,
                    "message": message,
                    "level": level,
                    "logger": logger_name
                })
                current_tool = None
                current_tool_args = None
                continue

            # Parse workflow steps
            step_match = re.search(step_pattern, message)
            if step_match:
                step_name, step_details = step_match.groups()
                current_step = step_name.strip()
                log_entries.append({
                    "timestamp": timestamp,
                    "type": "step",
                    "step": current_step,
                    "details": step_details.strip(),
                    "message": message,
                    "level": level,
                    "logger": logger_name
                })
                continue

            # Parse LLM prompts and responses
            if "LLM Prompt:" in message:
                # Extract the prompt text - it might span multiple lines
                current_llm_prompt = message.split("LLM Prompt:", 1)[1].strip()
                current_llm_id = f"llm_{len(llm_interactions)}"
                continue

            if "LLM Response: tokens=" in message:
                llm_response_match = re.search(llm_response_pattern, message)
                if llm_response_match and current_llm_prompt is not None:
                    total_tokens, prompt_tokens, completion_tokens = map(int, llm_response_match.groups())
                    current_llm_tokens = {
                        "total": total_tokens,
                        "prompt": prompt_tokens,
                        "completion": completion_tokens
                    }
                continue

            if "LLM Content:" in message and current_llm_prompt is not None:
                llm_content = message.split("LLM Content:", 1)[1].strip()

                # Create an LLM interaction entry
                llm_interaction = {
                    "timestamp": timestamp,
                    "type": "llm",
                    "id": current_llm_id,
                    "prompt": current_llm_prompt,
                    "response": llm_content,
                    "tokens": current_llm_tokens,
                    "logger": logger_name,
                    "level": level,
                    "current_step": current_step
                }

                llm_interactions.append(llm_interaction)
                log_entries.append(llm_interaction)

                # Reset LLM tracking variables
                current_llm_prompt = None
                current_llm_tokens = None
                current_llm_id = None
                continue

            # Add general log entry
            log_entries.append({
                "timestamp": timestamp,
                "type": "log",
                "message": message,
                "level": level,
                "logger": logger_name,
                "current_step": current_step
            })

    # Convert agents set to list for JSON serialization
    workflow_metadata["agents"] = list(workflow_metadata["agents"])

    return log_entries, workflow_metadata

# Parse the selected log file
log_entries, workflow_metadata = parse_log_file(selected_log_file)

# Display workflow metadata
st.header("Workflow Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Workflow ID", workflow_metadata["workflow_id"])
with col2:
    start_time = workflow_metadata.get("start_time", "Unknown")
    if isinstance(start_time, str) and start_time != "Unknown":
        start_time = start_time.replace("T", " ").split(".")[0]
    st.metric("Start Time", start_time)
with col3:
    status = workflow_metadata.get("status", "Unknown")
    st.metric("Status", status)

# Create a DataFrame from log entries
df = pd.DataFrame(log_entries)
if not df.empty:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

# Display workflow steps visualization
st.header("Workflow Steps")

# Filter for step entries
step_entries = [entry for entry in log_entries if entry["type"] == "step"]
if step_entries:
    # Create a timeline of steps
    step_df = pd.DataFrame(step_entries)
    step_df["timestamp"] = pd.to_datetime(step_df["timestamp"])

    # Add end times (estimated as the start of the next step or current time)
    step_df["end_time"] = step_df["timestamp"].shift(-1)
    step_df.loc[step_df["end_time"].isna(), "end_time"] = pd.Timestamp.now()

    # Create a Gantt chart
    fig = px.timeline(
        step_df,
        x_start="timestamp",
        x_end="end_time",
        y="step",
        color="step",
        hover_data=["details"],
        title="Workflow Steps Timeline"
    )

    # Add connecting lines between steps for visual clarity
    for i in range(len(step_df) - 1):
        fig.add_trace(
            go.Scatter(
                x=[step_df.iloc[i]["end_time"], step_df.iloc[i+1]["timestamp"]],
                y=[step_df.iloc[i]["step"], step_df.iloc[i+1]["step"]],
                mode="lines",
                line=dict(width=2, color="gray", dash="dot"),
                showlegend=False
            )
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Step",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display step details in a table
    st.subheader("Step Details")
    step_table = step_df[["step", "details", "timestamp"]].rename(
        columns={"timestamp": "Start Time"}
    )
    st.dataframe(step_table, use_container_width=True)
else:
    st.info("No workflow steps found in the log file.")

# Display timing information
st.header("Timing Analysis")

# Filter for timing entries
timing_entries = [entry for entry in log_entries if entry["type"] == "timing"]
if timing_entries:
    timing_df = pd.DataFrame(timing_entries)
    timing_df["timestamp"] = pd.to_datetime(timing_df["timestamp"])

    # Group by category and operation
    timing_summary = timing_df.groupby(["category", "operation"])["duration"].agg(
        ["count", "sum", "mean", "min", "max"]
    ).reset_index()

    # Create a bar chart of total duration by category
    category_totals = timing_df.groupby("category")["duration"].sum().reset_index()
    fig = px.bar(
        category_totals,
        x="category",
        y="duration",
        title="Total Duration by Category",
        labels={"duration": "Duration (seconds)", "category": "Category"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add end times for operations
    timing_df["end_time"] = timing_df.apply(lambda row: row["timestamp"] + pd.Timedelta(seconds=row["duration"]), axis=1)

    # Create a timeline of operations
    fig = px.timeline(
        timing_df,
        x_start="timestamp",
        x_end="end_time",
        y="operation",
        color="category",
        hover_data=["duration"],
        title="Operations Timeline"
    )

    # Add connecting lines between operations in the same category
    categories = timing_df["category"].unique()
    for category in categories:
        category_df = timing_df[timing_df["category"] == category].sort_values("timestamp")
        if len(category_df) > 1:
            for i in range(len(category_df) - 1):
                fig.add_trace(
                    go.Scatter(
                        x=[category_df.iloc[i]["end_time"], category_df.iloc[i+1]["timestamp"]],
                        y=[category_df.iloc[i]["operation"], category_df.iloc[i+1]["operation"]],
                        mode="lines",
                        line=dict(width=1, color="gray", dash="dot"),
                        showlegend=False
                    )
                )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Operation",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display timing summary in a table
    st.subheader("Timing Summary")
    timing_summary = timing_summary.sort_values("sum", ascending=False)
    timing_summary.columns = ["Category", "Operation", "Count", "Total (s)", "Mean (s)", "Min (s)", "Max (s)"]
    st.dataframe(timing_summary, use_container_width=True)
else:
    st.info("No timing information found in the log file.")

# Display tool calls
st.header("Tool Calls")

# Filter for tool entries
tool_entries = [entry for entry in log_entries if entry["type"] == "tool"]
if tool_entries:
    tool_df = pd.DataFrame(tool_entries)
    tool_df["timestamp"] = pd.to_datetime(tool_df["timestamp"])

    # Count tool usage
    tool_counts = tool_df["tool"].value_counts().reset_index()
    tool_counts.columns = ["Tool", "Count"]

    # Create a bar chart of tool usage
    fig = px.bar(
        tool_counts,
        x="Tool",
        y="Count",
        title="Tool Usage",
        labels={"Count": "Number of Calls", "Tool": "Tool Name"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display tool calls in a table
    st.subheader("Tool Call Details")
    tool_table = tool_df[["timestamp", "tool", "args"]].rename(
        columns={"timestamp": "Time", "tool": "Tool", "args": "Arguments"}
    )
    st.dataframe(tool_table, use_container_width=True)
else:
    st.info("No tool calls found in the log file.")

# Display LLM interactions
st.header("LLM Interactions")

# Filter for LLM entries
llm_entries = [entry for entry in log_entries if entry["type"] == "llm"]
if llm_entries:
    # Create a DataFrame for token usage
    llm_df = pd.DataFrame(llm_entries)
    llm_df["timestamp"] = pd.to_datetime(llm_df["timestamp"])

    # Extract token counts
    token_counts = []
    for entry in llm_entries:
        if entry.get("tokens"):
            token_counts.append({
                "id": entry["id"],
                "timestamp": entry["timestamp"],
                "prompt_tokens": entry["tokens"].get("prompt", 0),
                "completion_tokens": entry["tokens"].get("completion", 0),
                "total_tokens": entry["tokens"].get("total", 0)
            })

    token_df = pd.DataFrame(token_counts)
    if not token_df.empty:
        # Create a bar chart of token usage
        fig = px.bar(
            token_df,
            x="id",
            y=["prompt_tokens", "completion_tokens"],
            title="Token Usage by LLM Interaction",
            labels={"value": "Token Count", "variable": "Token Type", "id": "Interaction ID"},
            barmode="stack"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display LLM interactions
    for i, entry in enumerate(llm_entries):
        with st.expander(f"LLM Interaction {i+1} - {entry.get('timestamp').strftime('%H:%M:%S')} - {entry.get('logger', '')}"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Prompt")
                st.text_area(
                    "Prompt Text",
                    entry.get("prompt", ""),
                    height=300
                )

            with col2:
                st.subheader("Response")
                st.text_area(
                    "Response Text",
                    entry.get("response", ""),
                    height=300
                )

            # Display token information
            if entry.get("tokens"):
                st.metric(
                    "Total Tokens",
                    entry["tokens"].get("total", 0),
                    f"Prompt: {entry['tokens'].get('prompt', 0)} | Completion: {entry['tokens'].get('completion', 0)}"
                )

            # Display context information
            st.text(f"Step: {entry.get('current_step', 'Unknown')}")
            st.text(f"Logger: {entry.get('logger', 'Unknown')}")
            st.text(f"Timestamp: {entry.get('timestamp')}")
else:
    st.info("No LLM interactions found in the log file.")

# Display raw log entries
with st.expander("Raw Log Entries"):
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No log entries found.")

# This script is meant to be run with Streamlit
if __name__ == "__main__":
    # This block will only execute when run directly by Streamlit
    pass
